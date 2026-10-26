// Copyright 2026 Hyunho Cho
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// The pour law against a synthetic vessel. No ROS, no scale, no robot.
//
// The numbers the fixtures are built from are measurements, taken on the FR5
// cell's HS-AA with water on 2026-09-16:
//
//   5 Hz stream, 0.01 g resolution, 1.0 s to settle once the flow stops,
//   <= 0.21 g of tail at ~10 g/s, and -- the one that shapes the whole design --
//   at drip rates the indicator sat still and reported STABLE for 5.4 s with
//   material still arriving.
//
// The last one is why several tests below check that the planner never asks the
// scale whether it is done while the vessel is still tilted.
#include <algorithm>
#include <cmath>
#include <deque>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sdl_pour_controller/pour/flow_model.hpp"
#include "sdl_pour_controller/pour/material_profile.hpp"
#include "sdl_pour_controller/pour/pour_planner.hpp"
#include "sdl_pour_controller/pour/ramp.hpp"
#include "sdl_pour_controller/pour/shaker.hpp"
#include "sdl_pour_controller/pour/scale_filter.hpp"
#include "sdl_pour_controller/pour/shaping_law.hpp"
#include "sdl_pour_controller/pour/guards.hpp"
#include "pour_sim.hpp"

using sdl_pour_controller::pour::MaterialClass;
using sdl_pour_controller::pour::MaterialProfile;
using sdl_pour_controller::pour::PlannerConfig;
using sdl_pour_controller::pour::PourLimits;
using sdl_pour_controller::pour::PourObservation;
using sdl_pour_controller::pour::PourPhase;
using sdl_pour_controller::pour::PourPlanner;
using sdl_pour_controller::pour::PourRequest;
using sdl_pour_controller::pour::ScaleFilter;
using sdl_pour_controller::pour::ShapingConfig;
using sdl_pour_controller::pour::ShapingPourLaw;
using sdl_pour_controller::pour::pour_guard;
using pour_sim::RunOptions;
using pour_sim::RunResult;
using pour_sim::VesselSim;
using pour_sim::run_pour;

namespace {

PourLimits water_free()
{
    PourLimits l;
    l.max_flow_rate = 10.0;
    l.trim_flow_rate = 2.0;
    l.trim_pulse_sec = 0.15;
    l.transport_delay = 0.4;
    l.afterflow_grams = 0.3;
    l.settle_hold_sec = 1.2;
    l.tilt_rate = 0.5;
    l.seek_tilt_rate = 0.15;
    l.dose_quantum = 0.3;
    return l;
}

PourLimits honey_resistant()
{
    PourLimits l;
    l.max_flow_rate = 2.0;
    l.trim_flow_rate = 0.5;
    l.trim_pulse_sec = 0.4;
    l.transport_delay = 1.2;
    l.afterflow_grams = 4.0;
    l.settle_hold_sec = 3.0;
    l.tilt_rate = 0.2;
    l.seek_tilt_rate = 0.06;
    // The smallest amount a material this thick can be metered in is the string
    // it leaves behind every time the flow breaks, not something finer.
    l.dose_quantum = 3.0;
    return l;
}

//: honey_resistant() with what oligosaccharide syrup did on the rig
//: (2026-10-03): it kept pouring 1.2 g/s parked at its onset and dripped 10 deg
//: under it, so it is parked 0.3 rad further back, and with the stream stopped
//: there its tail is expected well under the 2.89 g a shallow park let go.
PourLimits syrup_resistant()
{
    PourLimits l = honey_resistant();
    l.park_drop = 0.3;
    l.afterflow_grams = 1.5;
    l.dose_quantum = 1.0;
    return l;
}

//: A vessel of syrup: starts at 46 deg, runs at ~1.5 g/s, and once running
//: keeps running until it is tipped back 0.25 rad under where it started.
RunOptions syrup_vessel()
{
    RunOptions opt;
    opt.sim_onset = 0.8;
    opt.sim_gain = 8.0;
    opt.sim_transport_delay = 1.0;
    opt.sim_tail = 0.5;
    opt.sim_drain_below = 0.25;
    return opt;
}

PourLimits salt_free()
{
    PourLimits l;
    l.max_flow_rate = 8.0;
    l.trim_flow_rate = 1.5;
    l.trim_pulse_sec = 0.2;
    l.transport_delay = 0.3;
    l.afterflow_grams = 0.5;
    l.settle_hold_sec = 1.5;
    l.tilt_rate = 0.35;
    l.seek_tilt_rate = 0.10;
    l.dose_quantum = 0.5;
    return l;
}

PourLimits damp_powder_resistant()
{
    PourLimits l;
    l.max_flow_rate = 3.0;
    l.trim_flow_rate = 0.8;
    l.trim_pulse_sec = 0.5;
    l.transport_delay = 0.6;
    l.afterflow_grams = 3.0;
    l.settle_hold_sec = 3.0;
    l.tilt_rate = 0.18;
    l.seek_tilt_rate = 0.05;
    l.dose_quantum = 2.0;
    return l;
}

MaterialProfile liquid_profile()
{
    MaterialProfile p;
    p.free = water_free();
    p.resistant = honey_resistant();
    return p;
}

MaterialProfile granular_profile()
{
    MaterialProfile p;
    p.free = salt_free();
    p.resistant = damp_powder_resistant();
    return p;
}


PourPlanner make_planner(PlannerConfig config = PlannerConfig{})
{
    PourPlanner planner;
    std::string why;
    EXPECT_TRUE(planner.configure(config, liquid_profile(), granular_profile(), why)) << why;
    return planner;
}

PourRequest water_request(double target = 50.0, double container = 139.15)
{
    PourRequest r;
    r.target_grams = target;
    r.container_grams = container;
    r.tolerance = 0.5;
    r.timeout = 180.0;
    r.max_tilt = 1.2;
    r.max_tilt_rate = 0.0;
    r.material = MaterialClass::Liquid;
    r.flow_index = 0.0;
    return r;
}

}  // namespace

// ---------------------------------------------------------------- profiles --

TEST(MaterialProfile, InterpolatesBetweenEndpointsAndClampsOutside)
{
    const auto profile = liquid_profile();
    EXPECT_DOUBLE_EQ(profile.at(0.0).max_flow_rate, water_free().max_flow_rate);
    EXPECT_DOUBLE_EQ(profile.at(1.0).max_flow_rate, honey_resistant().max_flow_rate);
    EXPECT_NEAR(profile.at(0.5).max_flow_rate, 6.0, 1e-9);
    // Past the endpoints nobody characterised the material, so the guess stops
    // growing rather than extrapolating into a tilt rate no one has seen.
    EXPECT_DOUBLE_EQ(profile.at(5.0).max_flow_rate, honey_resistant().max_flow_rate);
    EXPECT_DOUBLE_EQ(profile.at(-3.0).max_flow_rate, water_free().max_flow_rate);
}

TEST(MaterialProfile, ParkDropFollowsTheIndexAndMayNotBeNegative)
{
    MaterialProfile profile = liquid_profile();
    profile.resistant = syrup_resistant();
    EXPECT_DOUBLE_EQ(profile.at(0.0).park_drop, 0.0);
    EXPECT_NEAR(profile.at(0.5).park_drop, 0.15, 1e-12);
    EXPECT_DOUBLE_EQ(profile.at(1.0).park_drop, 0.3);
    std::string why;
    EXPECT_TRUE(profile.validate("liquid", why)) << why;
    profile.resistant.park_drop = -0.1;
    EXPECT_FALSE(profile.validate("liquid", why));
    EXPECT_NE(why.find("park_drop"), std::string::npos) << why;
}

TEST(MaterialProfile, NonFiniteIndexFallsBackToFreeRatherThanPropagating)
{
    const auto profile = liquid_profile();
    const auto limits = profile.at(std::nan(""));
    EXPECT_TRUE(std::isfinite(limits.tilt_rate));
    EXPECT_DOUBLE_EQ(limits.max_flow_rate, water_free().max_flow_rate);
}

TEST(MaterialProfile, RejectsATransposedPair)
{
    MaterialProfile swapped;
    swapped.free = honey_resistant();
    swapped.resistant = water_free();
    std::string why;
    EXPECT_FALSE(swapped.validate("liquid", why));
    EXPECT_NE(why.find("transposed"), std::string::npos) << why;
}

TEST(MaterialProfile, RejectsATrimRateFasterThanTheBulkRate)
{
    MaterialProfile p = liquid_profile();
    p.free.trim_flow_rate = p.free.max_flow_rate * 2.0;
    std::string why;
    EXPECT_FALSE(p.validate("liquid", why));
    EXPECT_NE(why.find("SLOWER"), std::string::npos) << why;
}

// ------------------------------------------------------------ scale filter --

TEST(ScaleFilter, RejectsAStepNoFlowCouldProduce)
{
    ScaleFilter filter;
    ScaleFilter::Config c;
    c.max_step_grams = 6.0;
    filter.configure(c);

    EXPECT_TRUE(filter.push({100.0, 0.0, true}));
    EXPECT_TRUE(filter.push({102.0, 0.2, false}));
    // The pan was knocked. 90 g in 200 ms is not a pour at any rate.
    EXPECT_FALSE(filter.push({192.0, 0.4, false}));
    EXPECT_EQ(filter.consecutive_rejects(), 1);
    // Nothing moved: the rejected sample did not become the new mass.
    EXPECT_DOUBLE_EQ(filter.grams(), 102.0);
    EXPECT_TRUE(filter.push({104.0, 0.6, false}));
    EXPECT_EQ(filter.consecutive_rejects(), 0);
}

TEST(ScaleFilter, RejectsNaNAndNonAdvancingStamps)
{
    ScaleFilter filter;
    ScaleFilter::Config c;
    c.max_step_grams = 6.0;
    filter.configure(c);

    EXPECT_TRUE(filter.push({100.0, 0.0, true}));
    EXPECT_FALSE(filter.push({std::nan(""), 0.2, true}));
    EXPECT_FALSE(filter.push({101.0, 0.0, true}));   // repeated stamp
    EXPECT_FALSE(filter.push({101.0, -0.1, true}));  // reordered
    EXPECT_TRUE(std::isfinite(filter.grams()));
    EXPECT_TRUE(std::isfinite(filter.flow_rate()));
}

TEST(ScaleFilter, FitsAFlowRateFromA5HzRamp)
{
    ScaleFilter filter;
    ScaleFilter::Config c;
    c.max_step_grams = 6.0;
    c.rate_window_sec = 0.6;
    filter.configure(c);

    // 9 g/s, quantised to 0.01 g the way the HS-AA reports it.
    for (int i = 0; i < 12; ++i) {
        const double t = 0.2 * i;
        const double grams = std::round((100.0 + 9.0 * t) / 0.01) * 0.01;
        filter.push({grams, t, false});
    }
    EXPECT_NEAR(filter.flow_rate(), 9.0, 0.2);
}

TEST(ScaleFilter, SettleNeedsBothTheStableFlagAndAQuietValue)
{
    ScaleFilter filter;
    ScaleFilter::Config c;
    c.max_step_grams = 6.0;
    filter.configure(c);

    filter.push({100.0, 0.0, true});
    filter.push({100.0, 0.2, true});
    EXPECT_FALSE(filter.settled(0.2, 1.2));   // quiet, but not for long enough
    filter.push({100.0, 1.4, true});
    EXPECT_TRUE(filter.settled(1.4, 1.2));
    // One flagged-unstable sample is enough to withdraw it, even unchanged.
    filter.push({100.0, 1.6, false});
    EXPECT_FALSE(filter.settled(1.6, 1.2));
}

TEST(ScaleFilterSettleLiesDuringDrips, DocumentedBlindSpot)
{
    // Reproduces the measurement that shaped the design: during a 1.5 g/s drip
    // the HS-AA held one value and flagged it stable for 5.4 s while material
    // was still arriving. settled() answers "yes" here and it is WRONG -- which
    // is exactly why PourPlanner only asks once the vessel is parked below the
    // onset angle, where no flow is possible by construction.
    ScaleFilter filter;
    ScaleFilter::Config c;
    c.max_step_grams = 6.0;
    filter.configure(c);

    for (int i = 0; i <= 27; ++i) {
        filter.push({358.13, 0.2 * i, true});
    }
    EXPECT_TRUE(filter.settled(5.4, 1.2));
    EXPECT_TRUE(filter.settled(5.4, 3.0));
    // No hold time rescues it: the gap between avalanches has no upper bound.
    EXPECT_TRUE(filter.settled(5.4, 5.0));
}

// ------------------------------------------------------------------ pours --

TEST(PourPlanner, RefusesAVesselThatIsNotTheOneDescribed)
{
    // The goal describes a 200 g vessel; the pan holds the 139.15 g flask. That
    // is the case an offset-only design pours anyway -- and it is not
    // hypothetical: on the rig the same flask read 139.15 g dry and 149.02 g
    // with the last pour's water still in it.
    auto planner = make_planner();
    PourRequest mismatched = water_request(50.0);
    mismatched.container_grams = 200.0;

    ScaleFilter filter;
    ScaleFilter::Config c;
    c.max_step_grams = 6.0;
    filter.configure(c);
    planner.begin(mismatched, 0.0);

    bool finished = false;
    std::string message;
    for (int i = 0; i < 200 && !finished; ++i) {
        const double t = 0.2 * i;
        filter.push({139.15, t, true});

        PourObservation obs;
        obs.now = t;
        obs.has_reading = true;
        obs.scale_fresh = true;
        obs.grams = 139.15;
        obs.flow_rate = 0.0;
        obs.settled = filter.settled(t, planner.settle_hold());
        obs.tilt = 0.0;

        const auto cmd = planner.update(obs);
        finished = cmd.finished;
        message = cmd.message;
        EXPECT_DOUBLE_EQ(cmd.tilt_rate, 0.0) << "nothing may tilt before the vessel checks out";
        if (finished) {
            EXPECT_FALSE(cmd.success);
        }
    }
    ASSERT_TRUE(finished) << "a mismatched vessel must end the goal, not wait forever";
    EXPECT_NE(message.find("vessel"), std::string::npos) << message;
}

TEST(PourPlanner, PoursWaterToTargetWithinTolerance)
{
    auto planner = make_planner();
    const auto result = run_pour(planner, water_request(50.0), RunOptions{});
    ASSERT_TRUE(result.finished) << "planner never terminated";
    EXPECT_TRUE(result.success) << result.message;
    EXPECT_NEAR(result.delivered, 50.0, 1.0);
}

TEST(PourPlanner, NeverJudgesDoneWhileTheVesselIsStillTilted)
{
    // The phase order is the safety property: every Settle is reached through a
    // Retract, never straight from Bulk or Trim.
    auto planner = make_planner();
    const auto result = run_pour(planner, water_request(50.0), RunOptions{});
    ASSERT_TRUE(result.finished);
    for (std::size_t i = 0; i < result.phases_seen.size(); ++i) {
        if (result.phases_seen[i] == PourPhase::Settle) {
            ASSERT_GT(i, 0u);
            EXPECT_EQ(result.phases_seen[i - 1], PourPhase::Retract)
                << "a settle was entered without parking the vessel first";
        }
    }
}

TEST(PourPlanner, HandlesAVesselGrippedOffSquare)
{
    // The beaker is handed over already leaning 0.10 rad toward the spout, so it
    // reaches its onset that much earlier. Nothing is configured for it: the
    // onset is found, and the park angle is allowed below the carried attitude.
    auto planner = make_planner();
    RunOptions opt;
    opt.grasp_tilt = 0.10;
    const auto result = run_pour(planner, water_request(50.0), opt);
    ASSERT_TRUE(result.finished) << "planner never terminated";
    EXPECT_TRUE(result.success) << result.message;
    EXPECT_NEAR(result.delivered, 50.0, 1.0);
}

TEST(PourPlanner, ATiltedGraspReachesItsOnsetSooner)
{
    auto square = make_planner();
    auto tilted = make_planner();
    RunOptions opt_tilted;
    opt_tilted.grasp_tilt = 0.10;

    const auto a = run_pour(square, water_request(50.0), RunOptions{});
    const auto b = run_pour(tilted, water_request(50.0), opt_tilted);
    ASSERT_TRUE(a.success) << a.message;
    ASSERT_TRUE(b.success) << b.message;
    EXPECT_LT(b.peak_tilt, a.peak_tilt)
        << "a vessel already leaning should not have to be tilted as far again";
}

TEST(PourPlanner, ThickLiquidStillLandsAndTakesLonger)
{
    auto thin = make_planner();
    auto thick = make_planner();

    PourRequest syrup = water_request(50.0);
    syrup.flow_index = 1.0;
    syrup.tolerance = 1.0;  // below the profile's quantum; the floor should win

    RunOptions thick_opt;
    thick_opt.sim_gain = 8.0;            // pours far less readily
    thick_opt.sim_transport_delay = 1.1; // and takes longer to arrive
    thick_opt.sim_tail = 3.0;            // with a much larger tail

    const auto fast = run_pour(thin, water_request(50.0), RunOptions{});
    const auto slow = run_pour(thick, syrup, thick_opt);
    ASSERT_TRUE(slow.finished) << "thick pour never terminated";
    EXPECT_TRUE(slow.success) << slow.message;
    EXPECT_NEAR(slow.delivered, 50.0, 3.0);
    EXPECT_GT(slow.seconds, fast.seconds);
}

TEST(PourPlanner, ParksAThickLiquidFarEnoughBackThatItStops)
{
    // What the rig did (2026-10-03, flow_index 0): stopped at 9.27 g, parked
    // 2 deg back, where the syrup kept running, and ended at 12.16 g.
    PourRequest syrup = water_request(10.0);
    syrup.flow_index = 1.0;
    syrup.tolerance = 1.0;

    PlannerConfig config;
    MaterialProfile with = liquid_profile();
    with.resistant = syrup_resistant();
    PourPlanner shallow, deep;
    std::string why;
    ASSERT_TRUE(shallow.configure(config, liquid_profile(), granular_profile(), why)) << why;
    ASSERT_TRUE(deep.configure(config, with, granular_profile(), why)) << why;

    const auto a = run_pour(shallow, syrup, syrup_vessel());
    const auto b = run_pour(deep, syrup, syrup_vessel());
    ASSERT_TRUE(b.finished) << "the deep-parked pour never terminated";
    EXPECT_TRUE(b.success) << b.message;
    EXPECT_NEAR(b.delivered, 10.0, 1.0);
    ASSERT_FALSE(b.park_angles.empty());
    for (const double park : b.park_angles) {
        EXPECT_LT(park, 0.8 - 0.25) << "parked where the syrup still runs";
    }
    // Parked at its onset the same syrup keeps coming.
    EXPECT_GT(a.delivered, b.delivered);
}

TEST(MaterialProfile, ForecastMinPairsFollowsTheIndexAndIsBounded)
{
    MaterialProfile profile = liquid_profile();
    profile.resistant = syrup_resistant();
    profile.resistant.forecast_min_pairs = 5.0;
    EXPECT_DOUBLE_EQ(profile.at(0.0).forecast_min_pairs, 0.0);
    EXPECT_DOUBLE_EQ(profile.at(1.0).forecast_min_pairs, 5.0);
    std::string why;
    EXPECT_TRUE(profile.validate("liquid", why)) << why;
    profile.resistant.forecast_min_pairs = -1.0;
    EXPECT_FALSE(profile.validate("liquid", why));
    EXPECT_NE(why.find("forecast_min_pairs"), std::string::npos) << why;
}

TEST(PourPlanner, AThickLiquidsFirstLumpDoesNotStopThePour)
{
    // The rig, 2026-10-03 and -04: syrup let go of what the lip held as one
    // lump, 0.82-0.86 g inside a 0.2 s sample; the flow model fitted on that
    // forecast ~9 g still to come, the pour stopped at 0.86 g and trimmed the
    // rest for 100 s. Here the first of the stream comes with a 0.85 g lump.
    PourRequest syrup = water_request(10.0);
    syrup.flow_index = 1.0;
    syrup.tolerance = 1.0;
    RunOptions vessel = syrup_vessel();
    vessel.onset_lump_grams = 0.85;

    MaterialProfile trusting = liquid_profile();
    trusting.resistant = syrup_resistant();
    trusting.resistant.afterflow_grams = 0.3;   // the rig's prior from 2026-10-04
    MaterialProfile waiting = trusting;
    waiting.resistant.forecast_min_pairs = 5.0;

    PourPlanner a, b;
    std::string why;
    ASSERT_TRUE(a.configure(PlannerConfig{}, trusting, granular_profile(), why)) << why;
    ASSERT_TRUE(b.configure(PlannerConfig{}, waiting, granular_profile(), why)) << why;
    const auto early = run_pour(a, syrup, vessel);
    const auto late = run_pour(b, syrup, vessel);
    std::printf("trusting the first pair: %.2f g, %d trims, %.0f s; waiting for 5: %.2f g, %d trims, %.0f s\n",
                early.delivered, early.trim_pulses, early.seconds,
                late.delivered, late.trim_pulses, late.seconds);
    ASSERT_TRUE(late.finished);
    EXPECT_TRUE(late.success) << late.message;
    EXPECT_NEAR(late.delivered, 10.0, 1.0);
    EXPECT_LE(late.trim_pulses, 1) << "the first stop still came at the lump";
    EXPECT_LT(late.trim_pulses, early.trim_pulses) << "the lump did not stop the trusting pour early either";
}

TEST(PourPlanner, ATrimPulseClimbsAThickLiquidsParkDropAtTheTiltRate)
{
    // Short of the target after the first stop, a pulse has to get from a park
    // 0.3 rad under the onset back to it. Crept, that would be 0.3 / (0.33 x
    // 0.06) = 15 s a pulse before anything pours; at the tilt rate, 1.5 s.
    PourRequest syrup = water_request(10.0);
    syrup.flow_index = 1.0;
    syrup.tolerance = 1.0;
    MaterialProfile with = liquid_profile();
    with.resistant = syrup_resistant();
    with.resistant.afterflow_grams = 4.0;  // a big tail prior: stops early, then trims
    PourPlanner planner;
    std::string why;
    ASSERT_TRUE(planner.configure(PlannerConfig{}, with, granular_profile(), why)) << why;
    const auto r = run_pour(planner, syrup, syrup_vessel());
    ASSERT_TRUE(r.finished);
    EXPECT_TRUE(r.success) << r.message;
    ASSERT_GE(r.trim_pulses, 1);
    const double trim_sec = r.phase_seconds.count(PourPhase::Trim) ? r.phase_seconds.at(PourPhase::Trim) : 0.0;
    EXPECT_LT(trim_sec / r.trim_pulses, 10.0) << trim_sec << " s over " << r.trim_pulses << " pulses";
}

TEST(PourPlanner, ASmallGoalSeeksItsOnsetSlower)
{
    MaterialProfile liquid = liquid_profile();
    liquid.free.slow_seek_below_grams = 20.0;
    std::string why;
    ASSERT_TRUE(liquid.validate("liquid", why)) << why;
    EXPECT_DOUBLE_EQ(liquid.at(1.0).slow_seek_below_grams, 0.0) << "the resistant endpoint left off";
    PourPlanner planner;
    ASSERT_TRUE(planner.configure(PlannerConfig{}, liquid, granular_profile(), why)) << why;
    const auto seek_rate_for = [&](double target) {
        planner.begin(water_request(target, 0.0), 0.0);
        PourObservation obs;
        obs.has_reading = true;
        obs.scale_fresh = true;
        obs.settled = true;
        planner.update(obs);  // verify -> seek
        obs.settled = false;
        obs.tilt = 0.5;
        obs.now = 0.1;
        return planner.update(obs).tilt_rate;
    };
    const double full = liquid.free.seek_tilt_rate;
    EXPECT_NEAR(seek_rate_for(10.0), 0.5 * full, 1e-12);
    EXPECT_NEAR(seek_rate_for(15.0), 0.75 * full, 1e-12);
    EXPECT_NEAR(seek_rate_for(30.0), full, 1e-12) << "a goal past the bound seeks as before";
    EXPECT_NEAR(seek_rate_for(4.0), 0.5 * full, 1e-12) << "floored at slow_seek_min_fraction";
    liquid.free.slow_seek_below_grams = -1.0;
    EXPECT_FALSE(liquid.validate("liquid", why));
}

TEST(PourPlanner, ASlowerSeekLetsASmallWaterGoalStopOnTarget)
{
    // The rig's water (2026-10-04): the stream reached the scale a second after
    // it left the lip, and every 10 g goal had over 3 g in the air at its stop.
    MaterialProfile fast = liquid_profile();
    fast.free.transport_delay = 1.0;
    MaterialProfile slow = fast;
    slow.free.slow_seek_below_grams = 20.0;
    RunOptions vessel;
    vessel.sim_transport_delay = 1.0;
    vessel.sim_onset = 0.6;
    vessel.sim_gain = 120.0;
    PlannerConfig config;
    config.seek_fast_until = 0.35;
    PourPlanner a, b;
    std::string why;
    ASSERT_TRUE(a.configure(config, fast, granular_profile(), why)) << why;
    ASSERT_TRUE(b.configure(config, slow, granular_profile(), why)) << why;
    const PourRequest ten = water_request(10.0, 0.0);
    const auto before = run_pour(a, ten, vessel);
    const auto after = run_pour(b, ten, vessel);
    std::printf("10 g of water, seek at full rate: %.2f g, %d trims, %.0f s; at half: %.2f g, %d trims, %.0f s\n",
                before.delivered, before.trim_pulses, before.seconds,
                after.delivered, after.trim_pulses, after.seconds);
    ASSERT_TRUE(after.finished);
    EXPECT_LE(std::abs(after.delivered - 10.0), std::abs(before.delivered - 10.0) + 1e-9);
}

TEST(MaterialProfile, EarlyFlightFactorAndKeepFlowingMarginAreBounded)
{
    MaterialProfile liquid = liquid_profile();
    std::string why;
    EXPECT_DOUBLE_EQ(liquid.at(0.0).early_flight_factor, 1.0) << "believed whole by default";
    EXPECT_DOUBLE_EQ(liquid.at(0.0).keep_flowing_margin, 0.0) << "off by default";
    liquid.free.early_flight_factor = 0.0;
    EXPECT_FALSE(liquid.validate("liquid", why));
    EXPECT_NE(why.find("early_flight_factor"), std::string::npos) << why;
    liquid.free.early_flight_factor = 0.4;
    liquid.free.keep_flowing_margin = -0.01;
    EXPECT_FALSE(liquid.validate("liquid", why));
    EXPECT_NE(why.find("keep_flowing_margin"), std::string::npos) << why;
    liquid.free.keep_flowing_margin = 0.01;
    EXPECT_TRUE(liquid.validate("liquid", why)) << why;
}

TEST(PourPlanner, WaterRushingOffAWetLipIsMeteredNotStoppedAndTrimmed)
{
    // The rig's water on 2026-10-04: the stream a second from the scale, the
    // lip wet from the step before, letting go at once well past the onset.
    MaterialProfile before = liquid_profile();
    before.free.transport_delay = 1.0;
    before.free.slow_seek_below_grams = 20.0;
    before.free.forecast_min_pairs = 5.0;
    MaterialProfile after = before;
    after.free.early_flight_factor = 0.4;
    after.free.keep_flowing_margin = 0.01;
    // A rush, not a stream: what the wet lip held lands as one lump, then the
    // stream runs on at the tilt's own rate.
    RunOptions vessel;
    vessel.sim_transport_delay = 1.0;
    vessel.sim_onset = 0.6;
    vessel.sim_gain = 60.0;
    vessel.onset_lump_grams = 1.5;
    PlannerConfig config;
    config.seek_fast_until = 0.35;
    PourRequest ten = water_request(10.0, 0.0);
    ten.tolerance = 1.0;
    PourPlanner a, b;
    std::string why;
    ASSERT_TRUE(a.configure(config, before, granular_profile(), why)) << why;
    ASSERT_TRUE(b.configure(config, after, granular_profile(), why)) << why;
    const auto ra = run_pour(a, ten, vessel);
    const auto rb = run_pour(b, ten, vessel);
    std::printf("water off a wet lip, 10 g: stopped on the rush %.2f g, %d trims, %.0f s; metered %.2f g, %d trims, %.0f s\n",
                ra.delivered, ra.trim_pulses, ra.seconds, rb.delivered, rb.trim_pulses, rb.seconds);
    ASSERT_TRUE(rb.finished);
    EXPECT_LE(rb.trim_pulses, ra.trim_pulses);
    EXPECT_LE(std::abs(rb.delivered - 10.0), std::abs(ra.delivered - 10.0) + 0.3);
}

TEST(PourPlanner, AWaterStreamThinningAsTheVesselEmptiesIsTakenUpBeforeItBreaks)
{
    // A real beaker: the onset climbs as it empties. Kept flowing, the tilt
    // follows the onset up; left to the lead-bounded climb the stream thins
    // out and breaks first.
    MaterialProfile plain = liquid_profile();
    plain.free.transport_delay = 1.0;
    plain.free.forecast_min_pairs = 5.0;
    plain.free.early_flight_factor = 0.4;
    MaterialProfile kept = plain;
    kept.free.keep_flowing_margin = 0.03;
    RunOptions vessel;
    vessel.sim_transport_delay = 1.0;
    vessel.sim_gain = 40.0;
    vessel.cylinder_radius_mm = 26.0;
    vessel.cylinder_height_mm = 70.0;
    vessel.contents_grams = 70.0;
    PlannerConfig config;
    config.seek_fast_until = 0.35;
    PourRequest ten = water_request(10.0, 0.0);
    ten.tolerance = 1.0;
    PourPlanner a, b;
    std::string why;
    ASSERT_TRUE(a.configure(config, plain, granular_profile(), why)) << why;
    ASSERT_TRUE(b.configure(config, kept, granular_profile(), why)) << why;
    const auto ra = run_pour(a, ten, vessel);
    const auto rb = run_pour(b, ten, vessel);
    const auto bulk = [](const RunResult & r) {
        return r.phase_seconds.count(PourPhase::Bulk) ? r.phase_seconds.at(PourPhase::Bulk) : 0.0;
    };
    std::printf("10 g of water from an emptying beaker: lead-bound %.2f g, %d trims, bulk %.1f s; kept flowing %.2f g, %d trims, bulk %.1f s\n",
                ra.delivered, ra.trim_pulses, bulk(ra), rb.delivered, rb.trim_pulses, bulk(rb));
    ASSERT_TRUE(rb.finished);
    EXPECT_TRUE(rb.success) << rb.message;
    EXPECT_LE(rb.trim_pulses, ra.trim_pulses);
}

TEST(MaterialProfile, StopForecastScaleAndMaxTrimPulsesAreBounded)
{
    MaterialProfile liquid = liquid_profile();
    std::string why;
    EXPECT_DOUBLE_EQ(liquid.at(0.0).stop_forecast_scale, 1.0);
    EXPECT_DOUBLE_EQ(liquid.at(0.0).max_trim_pulses, -1.0) << "the planner's by default";
    liquid.free.max_trim_pulses = 0.0;
    liquid.resistant.max_trim_pulses = -1.0;
    EXPECT_DOUBLE_EQ(liquid.at(0.2).max_trim_pulses, 0.0) << "the nearer endpoint's, not interpolated";
    EXPECT_DOUBLE_EQ(liquid.at(0.8).max_trim_pulses, -1.0);
    EXPECT_TRUE(liquid.validate("liquid", why)) << why;
    liquid.free.max_trim_pulses = 1.5;
    EXPECT_FALSE(liquid.validate("liquid", why));
    EXPECT_NE(why.find("max_trim_pulses"), std::string::npos) << why;
    liquid.free.max_trim_pulses = 0.0;
    liquid.free.stop_forecast_scale = 0.0;
    EXPECT_FALSE(liquid.validate("liquid", why));
    EXPECT_NE(why.find("stop_forecast_scale"), std::string::npos) << why;
}

TEST(PourPlanner, WaterWithNoTrimsEndsShortInsteadOfRestartingTheStream)
{
    // A big tail prior stops the first stop early; untrimmed, the pour ends
    // there, short, rather than tipping the stream up again.
    MaterialProfile untrimmed = liquid_profile();
    untrimmed.free.afterflow_grams = 4.0;
    MaterialProfile trimmed = untrimmed;
    untrimmed.free.max_trim_pulses = 0.0;
    PourPlanner a, b;
    std::string why;
    ASSERT_TRUE(a.configure(PlannerConfig{}, trimmed, granular_profile(), why)) << why;
    ASSERT_TRUE(b.configure(PlannerConfig{}, untrimmed, granular_profile(), why)) << why;
    PourRequest ten = water_request(10.0, 0.0);
    ten.tolerance = 1.0;
    const auto t = run_pour(a, ten, RunOptions{});
    const auto u = run_pour(b, ten, RunOptions{});
    ASSERT_TRUE(t.finished && u.finished);
    ASSERT_GE(t.trim_pulses, 1) << "the trimmed pour should have needed one";
    EXPECT_EQ(u.trim_pulses, 0);
    EXPECT_FALSE(u.success);
    EXPECT_LT(u.delivered, 10.0 - 1.0);
    EXPECT_EQ(std::count(u.phases_seen.begin(), u.phases_seen.end(), PourPhase::Trim), 0);
    EXPECT_NE(u.message.find("not trimmed"), std::string::npos) << u.message;
}

TEST(PourPlanner, AScaledDownForecastStopsLater)
{
    MaterialProfile whole = liquid_profile();
    whole.free.transport_delay = 1.0;
    whole.free.max_trim_pulses = 0.0;
    whole.free.afterflow_grams = 1.5;   // stops early at 1
    MaterialProfile scaled = whole;
    scaled.free.stop_forecast_scale = 0.5;
    PourPlanner a, b;
    std::string why;
    ASSERT_TRUE(a.configure(PlannerConfig{}, whole, granular_profile(), why)) << why;
    ASSERT_TRUE(b.configure(PlannerConfig{}, scaled, granular_profile(), why)) << why;
    PourRequest ten = water_request(10.0, 0.0);
    ten.tolerance = 1.0;
    RunOptions vessel;
    vessel.sim_transport_delay = 1.0;
    const auto ra = run_pour(a, ten, vessel);
    const auto rb = run_pour(b, ten, vessel);
    std::printf("10 g of water, forecast whole %.2f g, scaled by 0.5 %.2f g\n", ra.delivered, rb.delivered);
    EXPECT_GT(rb.delivered, ra.delivered);
    EXPECT_LT(ra.delivered, 10.0) << "the whole forecast should have stopped it short";
}

TEST(MaterialProfile, FinalApproachGramsIsBounded)
{
    MaterialProfile liquid = liquid_profile();
    std::string why;
    EXPECT_DOUBLE_EQ(liquid.at(0.0).final_approach_grams, 0.0) << "off by default";
    liquid.free.final_approach_grams = -1.0;
    EXPECT_FALSE(liquid.validate("liquid", why));
    EXPECT_NE(why.find("final_approach_grams"), std::string::npos) << why;
    liquid.free.final_approach_grams = 3.0;
    EXPECT_TRUE(liquid.validate("liquid", why)) << why;
}

// The rig's water as it poured on 2026-10-04 evening, and a beaker that
// empties, so the stream thins at a held tilt.
MaterialProfile rig_water_20261004()
{
    MaterialProfile p = liquid_profile();
    p.free.transport_delay = 1.0;
    p.free.tilt_rate = 0.15;
    p.free.seek_tilt_rate = 0.03;
    p.free.slow_seek_below_grams = 20.0;
    p.free.forecast_min_pairs = 5.0;
    p.free.early_flight_factor = 0.4;
    p.free.keep_flowing_margin = 0.03;
    p.free.stop_forecast_scale = 0.65;
    p.free.max_trim_pulses = 0.0;
    p.free.dose_quantum = 1.0;
    p.resistant.tilt_rate = 0.1;   // the rig's syrup, kept the slower end
    p.resistant.seek_tilt_rate = 0.02;
    return p;
}

RunOptions emptying_beaker()
{
    RunOptions vessel;
    vessel.sim_transport_delay = 1.0;
    vessel.sim_gain = 40.0;
    vessel.cylinder_radius_mm = 26.0;
    vessel.cylinder_height_mm = 70.0;
    vessel.contents_grams = 70.0;
    return vessel;
}

TEST(PourPlanner, AWaterStreamThinningInTheLastGramsIsHeldNotClimbedAfter)
{
    MaterialProfile climbing = rig_water_20261004();
    MaterialProfile held = climbing;
    held.free.final_approach_grams = 3.0;
    RunOptions vessel = emptying_beaker();
    vessel.watch_last_grams = 3.0;
    PlannerConfig config;
    config.seek_fast_until = 0.35;
    PourRequest ten = water_request(10.0, 0.0);
    ten.tolerance = 1.0;
    PourPlanner a, b;
    std::string why;
    ASSERT_TRUE(a.configure(config, climbing, granular_profile(), why)) << why;
    ASSERT_TRUE(b.configure(config, held, granular_profile(), why)) << why;
    const auto ra = run_pour(a, ten, vessel);
    const auto rb = run_pour(b, ten, vessel);
    std::printf("10 g of water, last 3 g: climbed %.2f deg -> %.2f g; held %.2f deg -> %.2f g\n",
                ra.rise_near_target * 180.0 / M_PI, ra.delivered,
                rb.rise_near_target * 180.0 / M_PI, rb.delivered);
    ASSERT_TRUE(rb.finished) << rb.message;
    EXPECT_LE(rb.rise_near_target, 1e-3) << "the bulk tilted further in its last 3 g";
    EXPECT_TRUE(rb.success) << rb.message;
}

TEST(PourPlanner, TheFinalApproachFollowsTheOnsetUpAtItsRisePerGram)
{
    // Held flat, an emptying beaker's stream thins on its own; followed up at
    // the onset's rise per gram, the tilt creeps as the flow lands and never
    // more than that rise over what is left.
    MaterialProfile held = rig_water_20261004();
    held.free.final_approach_grams = 3.0;
    MaterialProfile following = held;
    following.free.final_approach_rise = 0.009;
    RunOptions vessel = emptying_beaker();
    vessel.watch_last_grams = 3.0;
    PlannerConfig config;
    config.seek_fast_until = 0.35;
    PourRequest ten = water_request(10.0, 0.0);
    ten.tolerance = 1.0;
    PourPlanner a, b;
    std::string why;
    ASSERT_TRUE(a.configure(config, held, granular_profile(), why)) << why;
    ASSERT_TRUE(b.configure(config, following, granular_profile(), why)) << why;
    const auto ra = run_pour(a, ten, vessel);
    const auto rb = run_pour(b, ten, vessel);
    std::printf("10 g of water, last 3 g: held %.2f deg -> %.2f g; followed at 0.009 rad/g %.2f deg -> %.2f g\n",
                ra.rise_near_target * 180.0 / M_PI, ra.delivered,
                rb.rise_near_target * 180.0 / M_PI, rb.delivered);
    ASSERT_TRUE(rb.finished) << rb.message;
    EXPECT_TRUE(rb.success) << rb.message;
    EXPECT_GT(rb.rise_near_target, ra.rise_near_target);
    // What the scale shows lags what has left by a delay: allow the grams in
    // the air at the trim rate on top of the window's own.
    EXPECT_LE(rb.rise_near_target, 0.009 * (3.0 + 2.0 * 1.0) + 1e-3);
}

TEST(MaterialProfile, FinalApproachRiseIsBounded)
{
    MaterialProfile liquid = liquid_profile();
    std::string why;
    EXPECT_DOUBLE_EQ(liquid.at(0.0).final_approach_rise, 0.0) << "held by default";
    liquid.free.final_approach_rise = -0.001;
    EXPECT_FALSE(liquid.validate("liquid", why));
    EXPECT_NE(why.find("final_approach_rise"), std::string::npos) << why;
    liquid.free.final_approach_rise = 0.009;
    EXPECT_TRUE(liquid.validate("liquid", why)) << why;
}

TEST(MaterialProfile, MaxStepDownIsBounded)
{
    MaterialProfile liquid = liquid_profile();
    std::string why;
    EXPECT_DOUBLE_EQ(liquid.at(0.0).max_step_down, -1.0) << "the law's lead by default";
    liquid.free.max_step_down = 0.0;
    EXPECT_DOUBLE_EQ(liquid.at(0.2).max_step_down, 0.0) << "the nearer endpoint's";
    EXPECT_DOUBLE_EQ(liquid.at(0.8).max_step_down, -1.0);
    EXPECT_TRUE(liquid.validate("liquid", why)) << why;
    liquid.free.max_step_down = -0.5;
    EXPECT_FALSE(liquid.validate("liquid", why));
    EXPECT_NE(why.find("max_step_down"), std::string::npos) << why;
}

TEST(PourPlanner, WaterThatMayNotStepBackHoldsItsTiltThroughTooMuchFlow)
{
    // A vessel that pours twice as readily as the prior: the climb overshoots
    // the wanted flow, and the law steps back -- unless the material may not.
    MaterialProfile stepping = rig_water_20261004();
    stepping.free.final_approach_grams = 3.0;
    stepping.free.final_approach_rise = 0.009;
    MaterialProfile holding = stepping;
    holding.free.max_step_down = 0.0;
    // The rig's onset (35 deg) and twice the prior's gain.
    RunOptions vessel;
    vessel.sim_transport_delay = 1.0;
    vessel.sim_onset = 0.6;
    vessel.sim_gain = 80.0;
    PlannerConfig config;
    config.seek_fast_until = 0.35;
    PourRequest fifty = water_request(50.0, 0.0);
    fifty.tolerance = 1.0;
    PourPlanner a, b;
    std::string why;
    ASSERT_TRUE(a.configure(config, stepping, granular_profile(), why)) << why;
    ASSERT_TRUE(b.configure(config, holding, granular_profile(), why)) << why;
    const auto ra = run_pour(a, fifty, vessel);
    const auto rb = run_pour(b, fifty, vessel);
    std::printf("50 g of water, a quick pourer: stepped back %.2f deg -> %.2f g; held %.2f deg -> %.2f g\n",
                ra.bulk_step_down * 180.0 / M_PI, ra.delivered,
                rb.bulk_step_down * 180.0 / M_PI, rb.delivered);
    ASSERT_TRUE(rb.finished) << rb.message;
    EXPECT_GT(ra.bulk_step_down, 0.005) << "the case should step back without the cap";
    EXPECT_LE(rb.bulk_step_down, 1e-6);
    // The sim's stream does not break when stepped back, so it cannot show
    // what holding saves; only that the forecast still stops it in time.
    EXPECT_TRUE(rb.success) << rb.message;
}

TEST(PourPlanner, AStreamStoppedOutrightInTheFinalApproachIsStillClimbedAfter)
{
    // The whole bulk inside the final approach: held, the emptying beaker's
    // stream stops outright, and the pour climbs after it instead of waiting
    // out its timeout.
    MaterialProfile held = rig_water_20261004();
    held.free.final_approach_grams = 100.0;
    PlannerConfig config;
    config.seek_fast_until = 0.35;
    PourRequest ten = water_request(10.0, 0.0);
    ten.tolerance = 1.0;
    PourPlanner planner;
    std::string why;
    ASSERT_TRUE(planner.configure(config, held, granular_profile(), why)) << why;
    const auto r = run_pour(planner, ten, emptying_beaker());
    std::printf("10 g of water held from the onset: %.2f g in %.0f s, peak %.1f deg\n",
                r.delivered, r.seconds, r.peak_tilt * 180.0 / M_PI);
    ASSERT_TRUE(r.finished) << "waited out the run";
    EXPECT_GE(r.delivered, 10.0 - 1.0) << r.message;
}

TEST(PourPlanner, TheFinalApproachClimbsNoFasterThanItsOwnRate)
{
    // The whole bulk inside the final approach, as in the case above: the
    // emptying beaker's stream stops outright and is crept after; with a
    // final-approach rate that creep goes no faster, and the pour still gets
    // there.
    MaterialProfile creeping = rig_water_20261004();
    creeping.free.final_approach_grams = 100.0;
    MaterialProfile slow = creeping;
    slow.free.final_approach_tilt_rate = 0.0035;
    // Watched from 2 g in: the cycle the bulk is entered on carries the seek's
    // own rate on.
    RunOptions vessel = emptying_beaker();
    vessel.watch_last_grams = 8.0;
    PlannerConfig config;
    config.seek_fast_until = 0.35;
    PourRequest ten = water_request(10.0, 0.0);
    ten.tolerance = 1.0;
    PourPlanner a, b;
    std::string why;
    ASSERT_TRUE(a.configure(config, creeping, granular_profile(), why)) << why;
    ASSERT_TRUE(b.configure(config, slow, granular_profile(), why)) << why;
    const auto ra = run_pour(a, ten, vessel);
    const auto rb = run_pour(b, ten, vessel);
    std::printf("10 g of water held from the onset: crept at up to %.2f deg/s -> %.2f g in %.0f s; "
                "at up to %.2f deg/s -> %.2f g in %.0f s\n",
                ra.max_rate_near_target * 180.0 / M_PI, ra.delivered, ra.seconds,
                rb.max_rate_near_target * 180.0 / M_PI, rb.delivered, rb.seconds);
    ASSERT_TRUE(rb.finished) << rb.message;
    EXPECT_GT(ra.max_rate_near_target, 0.0035) << "the case should creep faster without the rate";
    EXPECT_LE(rb.max_rate_near_target, 0.0035 + 1e-12);
    EXPECT_GE(rb.delivered, 10.0 - 1.0) << rb.message;
}

TEST(MaterialProfile, FinalApproachTiltRateIsBounded)
{
    MaterialProfile p = granular_profile();
    std::string why;
    EXPECT_DOUBLE_EQ(p.at(0.0).final_approach_tilt_rate, 0.0) << "off by default";
    p.free.final_approach_tilt_rate = -0.001;
    EXPECT_FALSE(p.validate("granular", why));
    EXPECT_NE(why.find("final_approach_tilt_rate"), std::string::npos) << why;
    p.free.final_approach_tilt_rate = 0.0035;
    EXPECT_TRUE(p.validate("granular", why)) << why;
}

TEST(PourPlanner, ParkDropLeavesWaterAsItWas)
{
    MaterialProfile with = liquid_profile();
    with.resistant = syrup_resistant();
    PourPlanner before, after;
    std::string why;
    ASSERT_TRUE(before.configure(PlannerConfig{}, liquid_profile(), granular_profile(), why));
    ASSERT_TRUE(after.configure(PlannerConfig{}, with, granular_profile(), why));
    const auto a = run_pour(before, water_request(50.0), RunOptions{});
    const auto b = run_pour(after, water_request(50.0), RunOptions{});
    EXPECT_DOUBLE_EQ(a.delivered, b.delivered);
    EXPECT_DOUBLE_EQ(a.seconds, b.seconds);
    EXPECT_EQ(a.trim_pulses, b.trim_pulses);
}

TEST(PourPlanner, GranularMediaLandsWithinItsOwnDoseQuantum)
{
    auto planner = make_planner();
    PourRequest sugar = water_request(40.0);
    sugar.material = MaterialClass::Granular;
    sugar.flow_index = 0.0;
    sugar.tolerance = 0.5;  // below the profile's quantum; the floor should win

    RunOptions opt;
    opt.sim_gain = 30.0;
    opt.sim_transport_delay = 0.3;
    opt.sim_tail = 0.5;

    const auto result = run_pour(planner, sugar, opt);
    ASSERT_TRUE(result.finished) << "planner never terminated";
    EXPECT_TRUE(result.success) << result.message;
    // Salt's configured quantum is 0.5 g, and the pour is allowed a couple of
    // them: the floor is what one avalanche weighs, not what the goal asked for.
    EXPECT_NEAR(result.delivered, 40.0, 1.5);
}

TEST(PourPlanner, AGranularStopMarginAppliesToGranularPoursOnly)
{
    PlannerConfig shared;
    shared.stop_margin_factor = 1.0;
    PlannerConfig split = shared;
    split.granular_stop_margin_factor = 0.05;

    // Water pours exactly as before whether or not the granular factor is set.
    auto water_shared = make_planner(shared);
    auto water_split = make_planner(split);
    const auto a = run_pour(water_shared, water_request(30.0), RunOptions{});
    const auto b = run_pour(water_split, water_request(30.0), RunOptions{});
    EXPECT_DOUBLE_EQ(a.delivered, b.delivered);
    EXPECT_EQ(a.trim_pulses, b.trim_pulses);

    // A granular pour is stopped on the smaller margin: later, so it delivers
    // more on its first stop and needs no more trim pulses to finish.
    PourRequest sugar = water_request(30.0);
    sugar.material = MaterialClass::Granular;
    RunOptions opt;
    opt.sim_gain = 30.0;
    opt.sim_transport_delay = 0.3;
    opt.sim_tail = 0.5;
    auto sugar_shared = make_planner(shared);
    auto sugar_split = make_planner(split);
    const auto c = run_pour(sugar_shared, sugar, opt);
    const auto d = run_pour(sugar_split, sugar, opt);
    ASSERT_TRUE(c.finished && d.finished) << "planner never terminated";
    EXPECT_NE(c.delivered, d.delivered);
    EXPECT_LE(d.trim_pulses, c.trim_pulses);
}

TEST(PourPlanner, TheDoseQuantumFloorsTheRequestedTolerance)
{
    auto planner = make_planner();
    PourRequest powder = water_request(40.0);
    powder.material = MaterialClass::Granular;
    powder.flow_index = 1.0;   // dose_quantum 2.0 g
    powder.tolerance = 0.05;   // not achievable by any law

    planner.begin(powder, 0.0);
    EXPECT_DOUBLE_EQ(planner.report().effective_tolerance, 2.0);
}

TEST(PourPlanner, ReportsOvershootInsteadOfChasingIt)
{
    auto planner = make_planner();
    PourRequest request = water_request(50.0);
    request.tolerance = 0.2;

    RunOptions opt;
    // The vessel pours at a rate the profile knows, but when the stream breaks
    // it keeps draining twenty times longer than water does -- the surprise a
    // flow_index of 0 cannot anticipate. The bulk phase stops in the right
    // place and the tail carries it past anyway.
    opt.sim_gain = 40.0;
    opt.sim_transport_delay = 0.4;
    opt.sim_tail = 6.0;

    const auto result = run_pour(planner, request, opt);
    ASSERT_TRUE(result.finished);
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.message.find("overpoured"), std::string::npos) << result.message;
    // And it stopped: the old law drove the wrist backwards through the carried
    // attitude on a negative error until it hit the tilt bound.
    EXPECT_LE(result.peak_tilt, request.max_tilt + 1e-6);
}

TEST(PourPlanner, ParksTheVesselWhenTheScaleGoesQuiet)
{
    auto planner = make_planner();
    RunOptions opt;
    opt.scale_dies_at = 6.0;

    const auto result = run_pour(planner, water_request(200.0), opt);
    ASSERT_TRUE(result.finished);
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.message.find("quiet"), std::string::npos) << result.message;
    EXPECT_EQ(result.last_phase, PourPhase::Done);
}

TEST(PourPlanner, GivesUpWhenTheTiltBoundProducesNoFlow)
{
    auto planner = make_planner();
    PourRequest request = water_request(50.0);
    request.max_tilt = 0.05;  // well below the vessel's 0.15 rad onset

    const auto result = run_pour(planner, request, RunOptions{});
    ASSERT_TRUE(result.finished);
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.message.find("nothing came out"), std::string::npos) << result.message;
}

TEST(PourPlanner, CancelParksTheVesselBeforeEnding)
{
    auto planner = make_planner();
    PourRequest request = water_request(500.0);

    VesselSim vessel;
    ScaleFilter filter;
    ScaleFilter::Config c;
    c.max_step_grams = 6.0;
    filter.configure(c);

    double now = 0.0, tilt = 0.0, next_sample = 0.0, last = request.container_grams;
    planner.begin(request, now);
    bool cancelled = false, finished = false;
    double tilt_at_cancel = 0.0;

    for (int i = 0; i < 125 * 120 && !finished; ++i) {
        vessel.step(now, 1.0 / 125.0, tilt);
        if (now >= next_sample) {
            next_sample += 0.2;
            const double g = std::round((request.container_grams + vessel.landed) / 0.01) * 0.01;
            filter.push({g, now, std::abs(g - last) < 1e-9});
            last = g;
        }
        PourObservation obs;
        obs.now = now;
        obs.has_reading = filter.has_sample();
        obs.scale_fresh = obs.has_reading && filter.age(now) <= 0.5;
        obs.grams = filter.grams();
        obs.flow_rate = filter.flow_rate();
        obs.settled = filter.settled(now, planner.settle_hold());
        obs.tilt = tilt;

        if (!cancelled && vessel.landed > 20.0) {
            planner.cancel();
            cancelled = true;
            tilt_at_cancel = tilt;
        }
        const auto cmd = planner.update(obs);
        finished = cmd.finished;
        if (finished) {
            EXPECT_FALSE(cmd.success);
            EXPECT_EQ(cmd.message, "cancelled");
        }
        tilt += cmd.tilt_rate / 125.0;
        now += 1.0 / 125.0;
    }
    ASSERT_TRUE(finished);
    EXPECT_GT(tilt_at_cancel, 0.0);
    // Back to the attitude it was carried in, not abandoned mid-tip.
    EXPECT_NEAR(tilt, 0.0, 0.01);
}

TEST(PourPlanner, ABrokenScaleStopsTheTiltAndSaysItJumpedRatherThanWentQuiet)
{
    // Still publishing, every sample rejected as impossible. Staleness is
    // judged from ACCEPTED samples, so this stops the pour exactly as silence
    // would -- and the report must say which it was. "The scale went quiet"
    // sends an operator to debug a driver that is working.
    auto planner = make_planner();
    planner.begin(water_request(50.0), 0.0);

    PourObservation obs;
    obs.now = 0.0;
    obs.has_reading = true;
    obs.scale_fresh = true;
    obs.grams = 139.15;
    obs.settled = true;
    obs.tilt = 0.0;
    planner.update(obs);   // verify passes

    obs.now = 1.0;
    obs.settled = false;
    obs.scale_fresh = false;
    obs.consecutive_rejects = 3;
    obs.last_rejected_step = 90.0;
    const auto cmd = planner.update(obs);
    EXPECT_DOUBLE_EQ(cmd.tilt_rate, 0.0);
    ASSERT_EQ(planner.phase(), PourPhase::Retract);

    obs.now = 1.1;
    const auto done = planner.update(obs);   // already at tilt 0: retract completes
    ASSERT_TRUE(done.finished);
    EXPECT_NE(done.message.find("jumped by 90"), std::string::npos) << done.message;
    EXPECT_EQ(done.message.find("quiet"), std::string::npos) << done.message;
}

// ------------------------------------------------- the back of the range --

TEST(PourPlanner, AParkThatNeverSettlesWalksDownLinearlyAndStopsAtTheBackBound)
{
    // Something other than the lip keeps adding mass to the pan, so no park
    // ever settles and the planner keeps concluding its park is too high. That
    // is the case that used to compound the margin -- 0.06, 0.12, 0.36, 1.44,
    // 7.2 rad -- and swing the wrist backwards to its joint limit.
    auto planner = make_planner();
    RunOptions opt;
    opt.leak_gps = 0.5;
    opt.leak_after = 5.0;
    opt.max_back_tilt = 0.3;

    const auto result = run_pour(planner, water_request(50.0), opt);
    ASSERT_TRUE(result.finished) << "a park that cannot settle must end the goal";
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.message.find("max_back_tilt"), std::string::npos) << result.message;
    EXPECT_GE(result.min_tilt, -opt.max_back_tilt - 1e-9)
        << "the vessel went further behind its carried attitude than allowed";

    ASSERT_GE(result.park_angles.size(), 2u);
    for (std::size_t i = 1; i < result.park_angles.size(); ++i) {
        const double gap = result.park_angles[i - 1] - result.park_angles[i];
        EXPECT_GT(gap, 0.0) << "each re-park must be lower than the last";
        // Linear: the n-th retry drops n base margins, and the base is 0.06 rad
        // for water. A compounding margin blows through this by the third.
        EXPECT_LE(gap, 0.06 * static_cast<double>(i) + 0.01) << "re-park " << i;
    }
}

TEST(PourPlanner, AZeroBackBoundNeverTurnsTheVesselBehindItsCarriedAttitude)
{
    auto planner = make_planner();
    RunOptions opt;
    opt.leak_gps = 0.5;
    opt.leak_after = 5.0;
    opt.max_back_tilt = 0.0;

    const auto result = run_pour(planner, water_request(50.0), opt);
    ASSERT_TRUE(result.finished);
    EXPECT_FALSE(result.success);
    EXPECT_GE(result.min_tilt, -1e-9);
}

TEST(PourPlanner, AParkBehindAJointLimitEndsTheGoalInsteadOfHoldingItOpen)
{
    // The arm cannot go below 0.14 rad, which is just short of where the
    // planner wants to park (onset 0.15 less the margin). A retract ignores
    // cancel, timeout and staleness -- it is what they would have asked for --
    // so without a deadline this goal would never finish.
    auto planner = make_planner();
    RunOptions opt;
    opt.tilt_floor = 0.14;
    opt.horizon = 120.0;

    const auto result = run_pour(planner, water_request(50.0), opt);
    ASSERT_TRUE(result.finished) << "the planner is still in " << to_string(result.last_phase)
                                 << " after " << result.seconds << " s";
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.message.find("could not reach"), std::string::npos) << result.message;
    EXPECT_LT(result.seconds, 60.0);
}

TEST(PourPlanner, AnUnreachableParkAfterACancelStillEnds)
{
    // Cancel asks for a retract to the carried attitude. If that, too, is
    // pinned, the deadline is still what ends the goal.
    auto planner = make_planner();
    PourRequest request = water_request(50.0);
    request.max_back_tilt = 0.3;
    planner.begin(request, 0.0);

    PourObservation obs;
    obs.has_reading = true;
    obs.scale_fresh = true;
    obs.grams = request.container_grams;
    obs.settled = true;
    obs.tilt = 0.0;
    planner.update(obs);                    // verify passes

    obs.now = 0.1;
    obs.settled = false;
    obs.tilt = 0.4;                         // mid-pour
    planner.cancel();
    planner.update(obs);
    ASSERT_EQ(planner.phase(), PourPhase::Retract);

    bool finished = false;
    std::string message;
    for (int i = 1; i < 125 * 30 && !finished; ++i) {
        obs.now = 0.1 + i / 125.0;
        obs.tilt = 0.4;                     // never moves
        const auto cmd = planner.update(obs);
        finished = cmd.finished;
        message = cmd.message;
    }
    ASSERT_TRUE(finished);
    EXPECT_NE(message.find("could not reach"), std::string::npos) << message;
    EXPECT_NE(message.find("cancelled"), std::string::npos)
        << "the original reason must survive into the report: " << message;
}

// ------------------------------------------------------------- the guard --

TEST(PourGuard, SilenceAndImpossibleReadingsAreReportedDifferently)
{
    PourObservation obs;
    obs.now = 5.0;
    obs.has_reading = true;
    obs.scale_fresh = false;

    const std::string silent = pour_guard(obs, PourPhase::Bulk, false, 0.0, 60.0);
    EXPECT_NE(silent.find("quiet"), std::string::npos) << silent;

    obs.consecutive_rejects = 2;
    obs.last_rejected_step = -139.1;
    const std::string jumped = pour_guard(obs, PourPhase::Bulk, false, 0.0, 60.0);
    EXPECT_NE(jumped.find("jumped by -139.1"), std::string::npos) << jumped;
}

TEST(PourGuard, RetractIsNeverInterruptedAndVerifyRunsItsOwnDeadline)
{
    PourObservation obs;
    obs.now = 500.0;
    obs.scale_fresh = false;
    // Every reason to stop at once, and none of them applies to a retract:
    // a retract is what each of them would have asked for.
    EXPECT_EQ(pour_guard(obs, PourPhase::Retract, true, 0.0, 60.0), "");
    // Verify has not tilted; staleness there is the container gate's business.
    EXPECT_EQ(pour_guard(obs, PourPhase::Verify, false, 0.0, 0.0), "");
    EXPECT_EQ(pour_guard(obs, PourPhase::Verify, true, 0.0, 0.0), "cancelled");
}

// ------------------------------------------------------- the shaping law --

namespace {

ShapingConfig tuned_shaping()
{
    // The gains pouring_controller.yaml ships for the shaping law: the ones a sweep at
    // 50 g against pour_sim.hpp settled on. See the yaml for why not 0.008.
    ShapingConfig c;
    c.kp = 0.0001;
    c.kd = 0.005;
    return c;
}

ShapingPourLaw make_shaping(const ShapingConfig & config = tuned_shaping())
{
    ShapingPourLaw law;
    std::string why;
    EXPECT_TRUE(law.configure(config, why)) << why;
    return law;
}

}  // namespace

TEST(ShapingPourLaw, TheKernelIsADecayingSinusoidThatReversesSign)
{
    // The oscillation is the whole point of the paper's law: the negative lobe
    // is what makes the pour pause and pull back the way a chemist's does.
    auto law = make_shaping();
    const auto & k = law.kernel();
    ASSERT_EQ(k.size(), static_cast<std::size_t>(2.5 / 0.04));
    EXPECT_DOUBLE_EQ(k[0], 0.0);
    EXPECT_GT(*std::max_element(k.begin(), k.end()), 0.0);
    EXPECT_LT(*std::min_element(k.begin(), k.end()), 0.0);
}

TEST(ShapingPourLaw, PoursWaterToTargetAtItsTunedGains)
{
    // Slow by nature -- ~230 s on this vessel, against the phase machine's ~14 --
    // so it gets a timeout that measures accuracy rather than patience.
    auto law = make_shaping();
    PourRequest request = water_request(50.0);
    request.timeout = 400.0;
    RunOptions opt;
    opt.horizon = 500.0;
    const auto result = run_pour(law, request, opt);
    ASSERT_TRUE(result.finished) << "shaping law never terminated";
    EXPECT_TRUE(result.success) << result.message;
    EXPECT_NEAR(result.delivered, 50.0, 1.5);
}

TEST(ShapingPourLaw, StopsAtOrPastTheTargetInsteadOfChasingAnOvershoot)
{
    // A band test (|error| < tolerance) is jumped clean over when one sample
    // carries more than twice the tolerance, and the law then sees a negative
    // error and turns the wrist backwards after grams that cannot come back.
    // The shipped gains pour fast enough to do exactly that.
    //
    // At kp 0.002 against this vessel the band version ran to the 120 s timeout,
    // swinging the wrist back to the -0.3 rad bound; stopped at or past the
    // target it ends in ~12 s having never tilted behind upright.
    ShapingConfig eager = tuned_shaping();
    eager.kp = 0.002;
    eager.kd = 0.002;
    auto law = make_shaping(eager);
    PourRequest request = water_request(50.0);
    request.tolerance = 0.2;
    request.timeout = 120.0;

    RunOptions opt;
    opt.horizon = 200.0;
    const auto result = run_pour(law, request, opt);
    ASSERT_TRUE(result.finished);
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.message.find("overpoured"), std::string::npos) << result.message;
    // Never swung behind the attitude it was carried in -- beyond the one
    // control step a retract may land past zero inside its epsilon. The band
    // version reached the -0.3 rad bound.
    EXPECT_GE(result.min_tilt, -0.005);
    EXPECT_LT(result.seconds, 60.0) << "an overshoot must end the goal, not run to the timeout";
}

TEST(ShapingPourLaw, SharesThePhaseMachinesGuards)
{
    // Same dead scale, same stop, same words -- the A/B would otherwise be
    // comparing two different safety envelopes.
    auto law = make_shaping();
    auto planner = make_planner();
    RunOptions opt;
    opt.scale_dies_at = 6.0;

    const auto a = run_pour(law, water_request(200.0), opt);
    const auto b = run_pour(planner, water_request(200.0), opt);
    ASSERT_TRUE(a.finished);
    ASSERT_TRUE(b.finished);
    EXPECT_FALSE(a.success);
    EXPECT_NE(a.message.find("quiet"), std::string::npos) << a.message;
    EXPECT_NE(b.message.find("quiet"), std::string::npos) << b.message;
}

TEST(ShapingPourLaw, RefusesTheVesselTheGoalDidNotDescribe)
{
    // The container gate is shared too: the shaping law must not pour into a
    // vessel the phase machine would have refused.
    auto law = make_shaping();
    PourRequest request = water_request(50.0);
    request.container_grams = 200.0;
    RunOptions opt;
    opt.vessel_grams = 139.15;
    const auto result = run_pour(law, request, opt);
    ASSERT_TRUE(result.finished);
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.message.find("vessel"), std::string::npos) << result.message;
    EXPECT_DOUBLE_EQ(result.peak_tilt, 0.0);
}

TEST(ShapingPourLaw, TheGainsThisControllerShippedWithOutrunTheScaleAt50g)
{
    // pouring_controller.yaml says why the shaping law does not use 0.008 / 0.002: the
    // kernel's DC gain is ~1.25, so a 50 g error asks for the full tilt rate,
    // and the flow outruns anything the outlier bound accepts. Pinned here so
    // that the claim in the yaml stays true, and so that the pour is seen to
    // stop -- and to say why -- rather than keep tilting.
    ShapingConfig shipped;
    shipped.kp = 0.008;
    shipped.kd = 0.002;
    auto law = make_shaping(shipped);
    const auto result = run_pour(law, water_request(50.0), RunOptions{});
    ASSERT_TRUE(result.finished);
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.message.find("jumped"), std::string::npos) << result.message;
}

// ------------------------------------------- a vessel whose onset moves --

namespace {

// The FR5 bench's 100 mL beaker: 50 mm bore, 70 mm to the rim, filled to the
// 100 mL mark. Its onset is 0.65 rad full and 0.94 rad after 30 g has gone.
RunOptions hundred_ml_beaker()
{
    RunOptions o;
    o.cylinder_radius_mm = 25.0;
    o.cylinder_height_mm = 70.0;
    o.contents_grams = 100.0;
    o.horizon = 200.0;
    return o;
}

}  // namespace

TEST(PourPlanner, FinishesAPourFromABeakerWhoseOnsetRisesAsItEmpties)
{
    // Trim pulses used to aim at the onset the seek found. Over a 30 g pour
    // this beaker's onset rises 0.29 rad, so every pulse aimed at the old one
    // poured nothing: 8 pulses, 2.74 g short. They now creep up to wherever the
    // flow actually starts.
    for (double target : {30.0, 50.0}) {
        auto planner = make_planner();
        const auto result = run_pour(planner, water_request(target), hundred_ml_beaker());
        ASSERT_TRUE(result.finished) << target << " g never terminated";
        EXPECT_TRUE(result.success) << target << " g: " << result.message;
        EXPECT_NEAR(result.delivered, target, 0.5) << target << " g";
        EXPECT_LE(result.trim_pulses, 4) << target << " g";
    }
}

TEST(PourPlanner, EndsCleanlyWhenWhatIsLeftCannotReachTheLip)
{
    // max_tilt 1.2 rad cannot pour this beaker below ~33 g. Asking for 80 g of
    // its 100 must end, short, and say so -- not creep at the bound forever.
    auto planner = make_planner();
    const auto result = run_pour(planner, water_request(80.0), hundred_ml_beaker());
    ASSERT_TRUE(result.finished);
    EXPECT_FALSE(result.success);
    EXPECT_LT(result.delivered, 80.0);
    EXPECT_NE(result.message.find("tilt bound"), std::string::npos) << result.message;
    // It used to wait out the 180 s goal timeout while the last grams trickled.
    EXPECT_LT(result.seconds, 60.0);
}

// ------------------------------------------------- a stream that thins --

TEST(PourPlanner, AThinningStreamRaisesTheTiltNoFasterThanTheScaleCanAnswer)
{
    // The rig, 2026-09-24, 10 g of water: the stream thinned to 0.2 g/s at a
    // fixed tilt 5.2 g in. The bulk phase then commanded the angle its fitted
    // flow coefficient said would restore the flow -- and that coefficient, a
    // small flow over a small angle, had collapsed. The wrist tipped 9.5 deg in
    // 1.1 s with nothing on the scale to show for it, and 11 g landed after.
    //
    // Scripted rather than simulated, so the flow the planner sees is exactly
    // that one: whatever it does with the tilt, the scale keeps saying 0.2 g/s.
    auto planner = make_planner();
    planner.begin(water_request(50.0, 0.0), 0.0);

    constexpr double dt = 1.0 / 125.0;
    const double transport_delay = planner.limits().transport_delay;
    const double lead = PlannerConfig{}.max_tilt_lead;
    double tilt = 0.0;
    double grams = 0.0;
    double flow = 0.0;
    double bulk_since = -1.0;
    double worst_lead = 0.0;
    std::deque<std::pair<double, double>> history;  // (time, tilt) during the bulk

    for (double now = 0.0; now < 60.0; now += dt) {
        if (planner.phase() == PourPhase::Seek && tilt >= 0.5) {
            // The onset: a gram lands and the scale shows 2 g/s.
            grams = 1.0;
            flow = 2.0;
        }
        if (bulk_since >= 0.0 && now - bulk_since > 1.0) {
            // A second in, the stream thins, and the scale says so from then on.
            flow = 0.2;
        }
        if (bulk_since >= 0.0 && now - bulk_since > 10.0) {
            break;
        }

        PourObservation obs;
        obs.now = now;
        obs.has_reading = true;
        obs.scale_fresh = true;
        obs.settled = planner.phase() == PourPhase::Verify;
        obs.grams = grams;
        obs.flow_rate = flow;
        obs.tilt = tilt;

        const auto cmd = planner.update(obs);
        ASSERT_FALSE(cmd.finished) << cmd.message;
        if (cmd.phase == PourPhase::Bulk) {
            if (bulk_since < 0.0) {
                bulk_since = now;
            }
            history.emplace_back(now, tilt);
            while (history.size() > 1 && history.front().first < now - transport_delay) {
                history.pop_front();
            }
            worst_lead = std::max(worst_lead, tilt - history.front().second);
        } else {
            ASSERT_LT(bulk_since, 0.0) << "left the bulk phase for "
                                       << sdl_pour_controller::pour::to_string(cmd.phase);
        }
        tilt += cmd.tilt_rate * dt;
    }
    ASSERT_GE(bulk_since, 0.0) << "the seek never found the onset";
    // It may climb -- a thinning stream from an emptying vessel does need more
    // tilt -- but never by more than the scale has had time to answer for.
    EXPECT_LE(worst_lead, lead + 0.002)
        << "the bulk phase tipped " << worst_lead << " rad within one transport delay";
    EXPECT_GT(tilt, 0.5 + lead) << "a stream that thinned must still be followed up";
}

// ------------------------------------------------------------------ ramp --

TEST(Ramp, ReachesATargetAtRestWithoutOvershootOrAJump)
{
    // A retract from a creep: moving up at 0.0045 rad/s, told to go 0.07 rad
    // back at up to 0.15 rad/s. It used to reverse in one cycle.
    using sdl_pour_controller::pour::Ramp;
    constexpr double dt = 1.0 / 125.0;
    constexpr double accel = 0.4;
    Ramp ramp;
    ramp.reset(0.0045);
    double x = 0.80;
    const double target = 0.73;
    double v_prev = ramp.velocity();
    double lowest = x;
    int cycles = 0;
    while (x != target && cycles < 10000) {
        x += ramp.reach(x, target, 0.15, accel, dt);
        EXPECT_LE(std::abs(ramp.velocity()), 0.15 + 1e-12);
        if (x != target) {
            EXPECT_LE(std::abs(ramp.velocity() - v_prev), accel * dt + 1e-12)
                << "the speed jumped at cycle " << cycles;
        }
        v_prev = ramp.velocity();
        lowest = std::min(lowest, x);
        ++cycles;
    }
    EXPECT_EQ(x, target) << "never arrived";
    EXPECT_DOUBLE_EQ(ramp.velocity(), 0.0);
    EXPECT_GE(lowest, target - 1e-12) << "went past the park";
    // A trapezoid of 0.07 rad at 0.4 rad/s^2 with a 0.15 rad/s cap: ~0.84 s.
    EXPECT_LT(cycles * dt, 1.2);
}

TEST(Ramp, FollowsARateAtTheBoundedAccelerationAndCarriesOnFromWhatHappened)
{
    using sdl_pour_controller::pour::Ramp;
    constexpr double dt = 1.0 / 125.0;
    Ramp ramp;
    double step = 0.0;
    for (int i = 0; i < 10; ++i) {
        step = ramp.follow(0.15, 0.4, dt);
    }
    EXPECT_NEAR(ramp.velocity(), 10 * 0.4 * dt, 1e-12) << "reached the rate faster than allowed";
    EXPECT_NEAR(step, ramp.velocity() * dt, 1e-15);
    // Something downstream held it back to a third of the step: the next cycle
    // accelerates from that speed, not from the one it was handed.
    ramp.observed(step / 3.0, dt);
    EXPECT_NEAR(ramp.velocity(), step / 3.0 / dt, 1e-12);
    // Garbage in stops it rather than propagating.
    EXPECT_DOUBLE_EQ(ramp.follow(std::nan(""), 0.4, dt), 0.0);
    EXPECT_DOUBLE_EQ(ramp.velocity(), 0.0);
}

TEST(Ramp, WithAJerkBoundTheAccelerationIsContinuousToo)
{
    // The way back upright on 2026-10-03: from rest to 0.07 rad/s and down onto
    // a park 0.5 rad away, accel 0.2 rad/s^2, jerk 1 rad/s^3.
    using sdl_pour_controller::pour::Ramp;
    constexpr double dt = 1.0 / 125.0;
    constexpr double accel = 0.2;
    constexpr double jerk = 1.0;
    Ramp ramp;
    ramp.set_jerk(jerk);
    double x = 0.5;
    const double target = 0.0;
    double a_prev = 0.0;
    double v_prev = 0.0;
    double worst_jump = 0.0;
    int cycles = 0;
    while (x != target && cycles < 20000) {
        x += ramp.reach(x, target, 0.07, accel, dt);
        EXPECT_LE(std::abs(ramp.velocity()), 0.07 + 1e-9);
        EXPECT_LE(std::abs(ramp.acceleration()), accel + 1e-9);
        if (x != target) {
            worst_jump = std::max(worst_jump, std::abs(ramp.acceleration() - a_prev));
            EXPECT_LE(std::abs(ramp.velocity() - v_prev), accel * dt + 1e-9);
        }
        a_prev = ramp.acceleration();
        v_prev = ramp.velocity();
        EXPECT_GE(x, target - 1e-9) << "went past the park at cycle " << cycles;
        ++cycles;
    }
    EXPECT_EQ(x, target) << "never arrived";
    EXPECT_DOUBLE_EQ(ramp.velocity(), 0.0);
    // Shedding the acceleration inside the jerk bound, bar the last cycle that
    // lands on the rate: no corner of a trapezoid anywhere.
    EXPECT_LE(worst_jump, jerk * dt * 2.0 + 1e-9);
    // 0.5 rad at 0.07 rad/s is 7.1 s of cruise; the S-curves add well under 1 s.
    EXPECT_LT(cycles * dt, 8.5);
}

TEST(Ramp, AJerkBoundedFollowReachesTheRateWithoutRinging)
{
    using sdl_pour_controller::pour::Ramp;
    constexpr double dt = 1.0 / 125.0;
    Ramp ramp;
    ramp.set_jerk(1.0);
    double highest = 0.0;
    for (int i = 0; i < 400; ++i) {
        ramp.follow(0.1, 0.2, dt);
        highest = std::max(highest, ramp.velocity());
    }
    EXPECT_NEAR(ramp.velocity(), 0.1, 1e-9);
    EXPECT_LE(highest, 0.1 + 1e-9) << "overshot the rate";
    EXPECT_NEAR(ramp.acceleration(), 0.0, 1e-9);
    // Reversed: still S-shaped, and still never past the new rate.
    double lowest = 0.0;
    for (int i = 0; i < 600; ++i) {
        ramp.follow(-0.05, 0.2, dt);
        lowest = std::min(lowest, ramp.velocity());
    }
    EXPECT_NEAR(ramp.velocity(), -0.05, 1e-9);
    EXPECT_GE(lowest, -0.05 - 1e-9);
    // No bound set is the plain trapezoid, as before.
    Ramp plain;
    plain.follow(0.15, 0.4, dt);
    EXPECT_NEAR(plain.velocity(), 0.4 * dt, 1e-12);
}

TEST(PourPlanner, ARetractSaysWhereItIsGoingSoTheControllerCanBrakeIntoIt)
{
    // The controller brakes into the park only if the law names it; a retract
    // that only gave a rate would be stopped dead on the park, or run past it.
    auto probe = make_planner();
    probe.begin(water_request(20.0, 0.0), 0.0);
    PourObservation obs;
    obs.has_reading = true;
    obs.scale_fresh = true;
    obs.settled = true;
    probe.update(obs);  // verify -> seek
    obs.settled = false;
    obs.tilt = 0.4;
    obs.grams = 1.0;
    obs.flow_rate = 2.0;
    obs.now = 0.1;
    probe.update(obs);  // seek -> bulk
    obs.grams = 20.0;   // at the target: stops whatever the forecast says
    obs.now = 0.2;
    auto cmd = probe.update(obs);  // bulk -> retract
    ASSERT_EQ(cmd.phase, PourPhase::Retract);
    obs.now = 0.3;
    cmd = probe.update(obs);
    ASSERT_EQ(cmd.phase, PourPhase::Retract);
    EXPECT_TRUE(std::isfinite(cmd.target_tilt));
    EXPECT_LT(cmd.target_tilt, obs.tilt);
    EXPECT_LT(cmd.tilt_rate, 0.0);
}

// ------------------------------------------ the rig's delay, forecast --

namespace {

// The FR5 cell as the sim can play it, with the rig's own configuration as of
// 2026-09-28: a second from tilt to reading, the flow model seeded for its
// 100 mL beaker, and a bulk whose lead and flow scale with what remains.
PourPlanner make_rig_planner(double seek_fast_until = 0.0)
{
    MaterialProfile liquid;
    liquid.free = {8.0, 2.0, 0.15, 1.0, 0.3, 1.2, 0.15, 0.03, 0.3};
    liquid.resistant = {2.0, 0.5, 0.4, 1.2, 4.0, 3.0, 0.1, 0.02, 3.0};
    PlannerConfig config;
    config.max_tilt_lead = 0.01;
    config.max_tilt_lead_far = 0.05;
    config.lead_fraction = 0.15;
    config.trim_creep_fraction = 0.15;
    config.stop_margin_factor = 1.0;
    config.flow_model.prior_gain = 50.0;
    config.seek_fast_until = seek_fast_until;
    config.flow_model.prior_gain_sd = 150.0;
    config.flow_model.prior_rise = 0.006;
    config.flow_model.prior_rise_sd = 0.005;
    PourPlanner planner;
    std::string why;
    EXPECT_TRUE(planner.configure(config, liquid, liquid, why)) << why;
    return planner;
}

// A 100 mL beaker partly full, a 0.9 s pipe, and a lip that holds the water
// back past the onset and then lets go.
RunOptions rig_beaker(double gain, double sticky)
{
    RunOptions o;
    o.sim_transport_delay = 0.9;
    o.sim_gain = gain;
    o.cylinder_radius_mm = 25.0;
    o.cylinder_height_mm = 70.0;
    o.contents_grams = 70.0;
    o.sticky_excess = sticky;
    o.horizon = 240.0;
    return o;
}

}  // namespace

TEST(PourPlanner, ForecastingWhatIsInTheAirStopsTenGramsOnceWithTheRigsDelay)
{
    // With flow x delay x 1.5 for a stop margin, the rig's pours stopped the
    // bulk a fraction of a second in and left the rest to trim pulses, each of
    // which let 2.7-3.2 g off the lip at once: 11.27 and 12.61 g of 10. The
    // forecast lets the bulk run to the target and stop once.
    for (double gain : {120.0, 230.0, 350.0}) {
        for (double sticky : {0.0, 0.02}) {
            auto planner = make_rig_planner();
            const auto result = run_pour(planner, water_request(10.0), rig_beaker(gain, sticky));
            ASSERT_TRUE(result.finished) << gain << "/" << sticky;
            EXPECT_TRUE(result.success) << gain << "/" << sticky << ": " << result.message;
            EXPECT_NEAR(result.delivered, 10.0, 0.5) << gain << "/" << sticky;
            EXPECT_EQ(result.trim_pulses, 0) << gain << "/" << sticky;
        }
    }
}

TEST(PourPlanner, AReadingThatSpikesAndFallsBackDuringTheSeekIsNotTheOnset)
{
    // The rig, 2026-09-28, 50 g of water: 0 -> 0.99 -> 3.67 -> 0.20 g at 8.6
    // deg, far below where the beaker pours. The seek took the 0.99 g for the
    // onset; the pour ended 23.6 g short after eight trim pulses and 324 s.
    auto planner = make_rig_planner();
    RunOptions o = rig_beaker(230.0, 0.01);
    o.dip_at = 15.0;
    o.dip_grams = -3.5;  // up, not down
    o.dip_sec = 0.6;
    const auto result = run_pour(planner, water_request(20.0), o);
    ASSERT_TRUE(result.finished);
    EXPECT_TRUE(result.success) << result.message;
    EXPECT_NEAR(result.delivered, 20.0, 0.5);
    EXPECT_LE(result.trim_pulses, 1);
}

TEST(PourPlanner, ASmallSpikeDuringTheSeekThatFallsBackToNothingIsNotTheOnset)
{
    // The rig, 2026-10-05, syrup, a step from 40 to 50 g: the pan read +0.18 g
    // for 0.8 s at 23 deg, unstable, and fell back to where it was; the syrup
    // came at 56. Too small a fall for the fall-back above, it was taken for
    // the onset: the stop came on the first syrup, the park went 18 deg too
    // low, and four trims crept back up from it until the goal timed out.
    auto planner = make_rig_planner();
    RunOptions o = rig_beaker(230.0, 0.01);
    o.dip_at = 15.0;
    o.dip_grams = -0.18;  // up, not down
    o.dip_sec = 0.8;
    const auto result = run_pour(planner, water_request(10.0), o);
    ASSERT_TRUE(result.finished);
    EXPECT_TRUE(result.success) << result.message;
    EXPECT_NEAR(result.delivered, 10.0, 0.5);
    EXPECT_LE(result.trim_pulses, 1);
}

TEST(PourPlanner, AReadingThatDipsAndRecoversDuringTheSeekIsNotTheOnset)
{
    // The rig, 2026-09-28: 0.07 -> -0.22 -> -0.10 g during the seek fitted a
    // +0.17 g/s flow, the seek called the onset 14 deg early, the flow model
    // was seeded with it, and the pour stopped at a third of its target. In
    // the sim the same dip early in the seek ended pours 8 g short on average.
    auto planner = make_rig_planner();
    RunOptions o = rig_beaker(230.0, 0.01);
    o.dip_at = 15.0;
    o.dip_grams = 0.29;
    o.dip_sec = 1.0;
    const auto result = run_pour(planner, water_request(10.0), o);
    ASSERT_TRUE(result.finished);
    EXPECT_TRUE(result.success) << result.message;
    EXPECT_NEAR(result.delivered, 10.0, 0.5);
}

TEST(FlowModel, RecoversTheVesselFromPairsAndHoldsItsPriorWithoutThem)
{
    using sdl_pour_controller::pour::FlowModel;
    using sdl_pour_controller::pour::FlowModelConfig;
    FlowModelConfig c;
    c.prior_gain = 100.0;
    c.prior_gain_sd = 100.0;
    c.prior_onset_sd = 0.05;
    c.prior_rise = 0.0;
    c.prior_rise_sd = 0.02;
    c.flow_noise = 0.2;

    FlowModel model;
    model.begin(c, 0.80);
    // Nothing observed: the prior, exactly.
    EXPECT_NEAR(model.outflow(0.85, 0.0), 100.0 * 0.05, 1e-9);
    EXPECT_DOUBLE_EQ(model.outflow(0.70, 0.0), 0.0) << "below the onset nothing pours";

    // A vessel that pours 230 g/s per rad past 0.83, rising 0.006 rad per gram.
    const double gain = 230.0, onset = 0.83, rise = 0.006;
    for (int i = 0; i < 40; ++i) {
        const double left = 0.25 * i;
        const double tilt = onset + rise * left + 0.004 + 0.0005 * (i % 7);
        model.observe(tilt, left, gain * (tilt - onset - rise * left));
    }
    EXPECT_NEAR(model.gain(), gain, 0.15 * gain);
    EXPECT_NEAR(model.onset(0.0), onset, 0.01);
    EXPECT_NEAR(model.onset(10.0), onset + rise * 10.0, 0.01);
}

TEST(FlowModel, ASteadyClimbAtASteadyFlowDoesNotSlideTheOnsetAway)
{
    // The rig's 50 g of water, 2026-10-04 23:44: the tilt rose with what had
    // left and the flow held at ~3.3 g/s, so tilt and emptying could not be
    // told apart. The fit slid to a near-zero gain and an onset far below the
    // seek's, and forecast flow all the way down to the park: 10.6 g, when
    // 2.25 g came. These are that pour's pairs -- tilt a delay before [rad],
    // what had left [g], flow [g/s] -- as its log gives them.
    using sdl_pour_controller::pour::FlowModel;
    using sdl_pour_controller::pour::FlowModelConfig;
    static const double pairs[][3] = {
        {0.6093, 1.10, 3.02}, {0.6153, 0.64, 6.14}, {0.6213, 1.69, 5.07},
        {0.6273, 2.83, 3.47}, {0.6334, 3.48, 3.37}, {0.6394, 4.16, 2.90},
        {0.6452, 4.77, 2.99}, {0.6494, 5.32, 2.84}, {0.6494, 5.85, 2.73},
        {0.6518, 6.44, 2.68}, {0.6585, 7.00, 2.81}, {0.6663, 7.54, 2.95},
        {0.6752, 8.11, 2.94}, {0.6835, 8.70, 2.80}, {0.6892, 9.28, 2.88},
        {0.6943, 9.85, 3.05}, {0.7003, 10.46, 3.11}, {0.7070, 11.07, 3.12},
        {0.7143, 11.71, 3.18}, {0.7218, 12.33, 3.20}, {0.7283, 13.01, 3.03},
        {0.7347, 13.61, 3.17}, {0.7409, 14.25, 3.25}, {0.7474, 14.89, 3.28},
        {0.7544, 15.56, 3.34}, {0.7611, 16.21, 3.39}, {0.7679, 16.90, 3.43},
        {0.7744, 17.57, 3.46}, {0.7810, 18.27, 3.43}, {0.7875, 18.97, 3.51},
        {0.7943, 19.65, 3.47}, {0.8007, 20.31, 3.42}, {0.8069, 21.01, 3.25},
        {0.8127, 21.70, 3.38}, {0.8183, 22.33, 3.39}, {0.8236, 22.99, 3.34},
        {0.8289, 23.67, 3.20}, {0.8343, 24.31, 3.24}, {0.8394, 24.95, 3.22},
        {0.8444, 25.55, 3.13}, {0.8493, 26.22, 2.95}, {0.8540, 26.84, 3.10},
        {0.8589, 27.41, 3.10}, {0.8638, 28.01, 3.04}, {0.8689, 28.64, 2.98},
        {0.8740, 29.27, 3.14}, {0.8790, 29.86, 3.29}, {0.8842, 30.52, 3.31},
        {0.8895, 31.18, 3.27}, {0.8950, 31.82, 3.29}, {0.9008, 32.50, 3.40},
        {0.9067, 33.23, 3.62}, {0.9129, 33.93, 4.07}, {0.9194, 34.79, 4.07},
        {0.9266, 35.48, 3.31}, {0.9335, 36.41, 4.04}, {0.9402, 37.20, 4.37},
        {0.9469, 38.03, 4.04}, {0.9531, 38.76, 3.82}, {0.9576, 39.59, 3.55},
        {0.9577, 40.35, 3.77},
    };
    FlowModelConfig c;   // the rig's priors
    c.prior_gain = 50.0;
    c.prior_gain_sd = 150.0;
    c.prior_onset_sd = 0.05;
    c.prior_rise = 0.006;
    c.prior_rise_sd = 0.005;
    c.flow_noise = 0.5;
    FlowModelConfig off = c;
    off.max_onset_shift = 0.0;
    const double start = 0.6482;   // where the flow was first seen
    FlowModel guarded, free;
    guarded.begin(c, start);
    free.begin(off, start);
    for (const auto & p : pairs) {
        guarded.observe(p[0], p[1], p[2]);
        free.observe(p[0], p[1], p[2]);
    }
    const double left = 42.2;   // where it stopped
    std::printf("row 22's pairs: unguarded gain %.1f, onset at %.0f g %.1f deg; guarded gain %.1f, onset %.1f deg\n",
                free.gain(), left, free.onset(left) * 180.0 / M_PI, guarded.gain(),
                guarded.onset(left) * 180.0 / M_PI);
    EXPECT_LT(free.onset(0.0), start - 0.25) << "the unguarded fit should slide, as it did on the rig";
    EXPECT_LE(std::abs(guarded.onset(0.0) - start), 0.25 + 1e-9);
    EXPECT_GE(guarded.onset(left), guarded.onset(0.0)) << "the onset rose as it emptied";
    EXPECT_GT(guarded.gain(), 5.0);
    // At the stop (56 deg) the guarded fit has the flow dying out on the way
    // down within a few degrees, not 90.
    EXPECT_GT(guarded.onset(left), 0.80);
}

TEST(PourPlanner, TheBulkParksUnderTheOnsetItPouredAtNotTheOneItStartedAt)
{
    // The rig, 2026-09-28, 50 g: the onset climbed from 28 to ~50 deg as the
    // beaker emptied, the bulk stopped at 51 deg and parked at 24.7 -- below
    // the SEEK's onset -- and the one trim pulse then crept 107 s back up.
    auto planner = make_rig_planner();
    RunOptions o = rig_beaker(230.0, 0.02);
    o.contents_grams = 95.0;
    const auto result = run_pour(planner, water_request(50.0), o);
    ASSERT_TRUE(result.finished);
    ASSERT_FALSE(result.park_angles.empty());
    // Within a few degrees of the deepest tilt the bulk reached, not a
    // quarter of a radian under it.
    EXPECT_GT(result.park_angles.front(), result.peak_tilt - 0.12)
        << "parked at " << result.park_angles.front() << " after a peak of " << result.peak_tilt;
    EXPECT_NEAR(result.delivered, 50.0, 1.0);
}

TEST(Ramp, AThrottleToNothingBrakesInsteadOfStoppingDead)
{
    // The controller throttles a path's speed by how far the IK is behind; at
    // the far end of that the cap is zero, and a cap of zero used to set the
    // speed to zero in one cycle.
    using sdl_pour_controller::pour::Ramp;
    constexpr double dt = 1.0 / 125.0;
    Ramp ramp;
    ramp.reset(0.15);
    const double v0 = ramp.velocity();
    ramp.reach(0.0, 1.0, 0.0, 0.4, dt);
    EXPECT_NEAR(ramp.velocity(), v0 - 0.4 * dt, 1e-12);
}

TEST(Ramp, ABrakeDeceleratesHardButStillDeceleratesAndNeverReverses)
{
    // The IK falling behind used to reset() the path's ramp: the speed gone in
    // one cycle. A brake takes it off at the given rate, past the jerk bound,
    // and stops at rest.
    using sdl_pour_controller::pour::Ramp;
    constexpr double dt = 1.0 / 125.0;
    Ramp ramp;
    ramp.set_jerk(2.0);
    ramp.reset(-0.15);
    const double step = ramp.brake(1.6, dt);
    EXPECT_NEAR(ramp.velocity(), -0.15 + 1.6 * dt, 1e-12) << "the full rate at once: the jerk bound is set aside";
    EXPECT_NEAR(step, ramp.velocity() * dt, 1e-15);
    EXPECT_NEAR(ramp.acceleration(), 1.6, 1e-9);
    int cycles = 1;
    while (ramp.velocity() != 0.0 && cycles < 1000) {
        ramp.brake(1.6, dt);
        EXPECT_LE(ramp.velocity(), 0.0) << "reversed at cycle " << cycles;
        ++cycles;
    }
    EXPECT_EQ(ramp.velocity(), 0.0);
    EXPECT_NEAR(cycles * dt, 0.15 / 1.6, 2.0 * dt);
    EXPECT_EQ(ramp.brake(1.6, dt), 0.0) << "at rest it stays at rest";
}

TEST(PourPlanner, NothingPoursBelowTheFastSeekSoItIsCrossedAtTheTiltRate)
{
    // Searching for the onset from upright at the seek rate took 17-28 s of
    // every rig pour; below seek_fast_until nothing can pour, and the seek
    // crosses it at the material's tilt rate.
    auto slow = make_rig_planner(0.0);
    auto fast = make_rig_planner(0.35);
    const auto a = run_pour(slow, water_request(20.0), rig_beaker(230.0, 0.01));
    const auto b = run_pour(fast, water_request(20.0), rig_beaker(230.0, 0.01));
    ASSERT_TRUE(a.finished && b.finished);
    EXPECT_TRUE(b.success) << b.message;
    EXPECT_NEAR(b.delivered, 20.0, 0.5);
    EXPECT_LT(b.seconds, a.seconds - 8.0) << a.seconds << " s slow, " << b.seconds << " s fast";
}

// ------------------------------------------------------------------ shaking --

using sdl_pour_controller::pour::Shaker;

TEST(Shaker, ATapIsARaisedCosineAtTheRateBoundThatNeverUntips)
{
    // The sugar profile's taps on the FR5: 0.02 rad, four a second, with 80%
    // of j6's 0.625 rad/s per-cycle bound.
    constexpr double dt = 0.008;
    constexpr double amplitude = 0.02;
    constexpr double max_rate = 0.5;
    Shaker shaker;
    double previous = 0.0;
    double peak = 0.0;
    int pulses = 0;
    for (int i = 0; i < 125; ++i) {
        const double offset = shaker.step(amplitude, 0.25, max_rate, dt);
        EXPECT_GE(offset, 0.0);
        EXPECT_LE(offset, amplitude + 1e-12);
        EXPECT_LE(std::abs(offset - previous), max_rate * dt + 1e-12) << "cycle " << i;
        if (previous == 0.0 && offset > 0.0) {
            ++pulses;
        }
        peak = std::max(peak, offset);
        previous = offset;
    }
    EXPECT_EQ(pulses, 4);
    EXPECT_GT(peak, 0.95 * amplitude);
}

TEST(Shaker, SwitchedOffMidTapItComesBackDownAndStops)
{
    constexpr double dt = 0.008;
    Shaker shaker;
    for (int i = 0; i < 5; ++i) {
        shaker.step(0.02, 0.25, 0.5, dt);
    }
    ASSERT_GT(shaker.offset(), 0.0);
    ASSERT_FALSE(shaker.idle());
    double previous = shaker.offset();
    for (int i = 0; i < 125; ++i) {
        const double offset = shaker.step(0.0, 0.25, 0.5, dt);
        EXPECT_LE(std::abs(offset - previous), 0.5 * dt + 1e-12);
        previous = offset;
    }
    EXPECT_EQ(shaker.offset(), 0.0);
    EXPECT_TRUE(shaker.idle());
}

TEST(MaterialProfile, TheShakeAndReposeMayBeZeroButNotNegativeOrHuge)
{
    std::string why;
    MaterialProfile p = granular_profile();
    EXPECT_TRUE(p.validate("granular", why)) << why;
    p.free.shake_amplitude = 0.02;
    p.free.shake_period = 0.25;
    p.resistant.shake_amplitude = 0.04;
    p.resistant.shake_period = 0.15;
    p.free.repose_angle = 0.5;
    EXPECT_TRUE(p.validate("granular", why)) << why;
    EXPECT_NEAR(p.at(0.5).shake_amplitude, 0.03, 1e-12);
    EXPECT_NEAR(p.at(0.5).shake_period, 0.2, 1e-12);
    EXPECT_NEAR(p.at(0.5).repose_angle, 0.25, 1e-12);

    MaterialProfile bad = p;
    bad.free.shake_period = 0.0;  // a tap with no period
    EXPECT_FALSE(bad.validate("granular", why));
    bad = p;
    bad.resistant.shake_amplitude = 0.3;  // the lip would swing with it
    EXPECT_FALSE(bad.validate("granular", why));
    bad = p;
    bad.free.repose_angle = -0.1;
    EXPECT_FALSE(bad.validate("granular", why));
}

namespace {

MaterialProfile tapped_granular(double repose)
{
    MaterialProfile p = granular_profile();
    for (PourLimits * l : {&p.free, &p.resistant}) {
        l->shake_amplitude = 0.02;
        l->shake_period = 0.25;
        l->repose_angle = repose;
    }
    return p;
}

PourPlanner make_tapping_planner(double repose, double seek_fast_until = 0.0)
{
    PlannerConfig config;
    config.seek_fast_until = seek_fast_until;
    PourPlanner planner;
    std::string why;
    EXPECT_TRUE(planner.configure(config, liquid_profile(), tapped_granular(repose), why)) << why;
    return planner;
}

double shook_in(const RunResult & r, PourPhase phase)
{
    const auto it = r.shake_seconds.find(phase);
    return it == r.shake_seconds.end() ? 0.0 : it->second;
}

}  // namespace

TEST(PourPlanner, AGranularFinalApproachTapsWithItsOwnTapsAndOnlyThere)
{
    // The last grams tapped larger and slower than the bulk, and only those:
    // the bulk keeps the ordinary taps, and nothing taps once it has stopped.
    // The sim's bed runs on smoothly rather than sticking and slipping, so the
    // trim rate is set high enough that its last grams count as a trickle.
    PlannerConfig config;
    config.granular_stop_margin_factor = 0.05;   // the rig's: the bulk runs into its last grams
    MaterialProfile ordinary = tapped_granular(0.0);
    for (PourLimits * l : {&ordinary.free, &ordinary.resistant}) {
        l->trim_flow_rate = 4.0;
        l->max_flow_rate = std::max(l->max_flow_rate, 4.0);
    }
    MaterialProfile finale = ordinary;
    for (PourLimits * l : {&finale.free, &finale.resistant}) {
        l->final_approach_grams = 3.0;
        l->final_approach_shake_amplitude = 0.03;
        l->final_approach_shake_period = 0.25;
    }
    PourPlanner a, b;
    std::string why;
    ASSERT_TRUE(a.configure(config, liquid_profile(), ordinary, why)) << why;
    ASSERT_TRUE(b.configure(config, liquid_profile(), finale, why)) << why;
    PourRequest sugar = water_request(40.0);
    sugar.material = MaterialClass::Granular;
    RunOptions opt;
    opt.sim_gain = 6.0;
    opt.sim_transport_delay = 0.3;
    opt.sim_tail = 0.5;
    const auto ra = run_pour(a, sugar, opt);
    const auto rb = run_pour(b, sugar, opt);
    std::printf("granular 40 g: ordinary taps %.2f g; final-approach taps %.2f g\n", ra.delivered, rb.delivered);
    ASSERT_TRUE(ra.finished && rb.finished);
    EXPECT_NEAR(ra.max_shake_amplitude, 0.02, 1e-12);
    EXPECT_NEAR(rb.max_shake_amplitude, 0.03, 1e-12) << "the final approach's taps";
    EXPECT_EQ(shook_in(rb, PourPhase::Retract), 0.0);
    EXPECT_EQ(shook_in(rb, PourPhase::Settle), 0.0);
}

TEST(MaterialProfile, FinalApproachTapsAreBounded)
{
    MaterialProfile p = granular_profile();
    std::string why;
    EXPECT_DOUBLE_EQ(p.at(0.0).final_approach_shake_amplitude, 0.0) << "off by default";
    p.free.final_approach_shake_amplitude = 0.03;
    p.free.final_approach_shake_period = 0.0;
    EXPECT_FALSE(p.validate("granular", why));
    EXPECT_NE(why.find("final_approach_shake_period"), std::string::npos) << why;
    p.free.final_approach_shake_period = 0.25;
    EXPECT_TRUE(p.validate("granular", why)) << why;
    p.free.final_approach_shake_amplitude = 0.2;
    EXPECT_FALSE(p.validate("granular", why));
}

TEST(PourPlanner, OnlyAGranularProfileWithAShakeTapsAndNeverOnceItHasStopped)
{
    auto planner = make_tapping_planner(0.0);
    PourRequest sugar = water_request(40.0);
    sugar.material = MaterialClass::Granular;
    RunOptions opt;
    opt.sim_gain = 30.0;
    opt.sim_transport_delay = 0.3;
    opt.sim_tail = 0.5;
    const auto tapped = run_pour(planner, sugar, opt);
    ASSERT_TRUE(tapped.finished);
    EXPECT_TRUE(tapped.success) << tapped.message;
    EXPECT_GT(shook_in(tapped, PourPhase::Seek), 0.0);
    EXPECT_DOUBLE_EQ(tapped.max_shake_amplitude, 0.02);
    // Tapping a vessel that is meant to have stopped pours into the settle
    // the target is judged on.
    EXPECT_EQ(shook_in(tapped, PourPhase::Retract), 0.0);
    EXPECT_EQ(shook_in(tapped, PourPhase::Settle), 0.0);
    EXPECT_EQ(shook_in(tapped, PourPhase::Done), 0.0);

    // The same planner pouring water never asks for one.
    const auto water = run_pour(planner, water_request(40.0), RunOptions{});
    ASSERT_TRUE(water.finished);
    EXPECT_TRUE(water.shake_seconds.empty());
}

TEST(PourPlanner, AnArmThatCannotReachTheBoundEndsTheSeekThereInsteadOfWaiting)
{
    // The rig, 2026-09-28: sugar asked for up to 1.75 rad, j4 ran into its
    // limit at 1.354, and the seek sat there for 140 s -- the bound was never
    // reached, so no stall rule applied -- until the goal was cancelled.
    auto planner = make_planner();
    PourRequest request = water_request(20.0);
    request.max_tilt = 1.2;
    RunOptions opt;
    opt.sim_onset = 0.8;  // past what the arm can do
    opt.reach = 0.5;
    const auto result = run_pour(planner, request, opt);
    ASSERT_TRUE(result.finished) << "still waiting at " << result.seconds << " s";
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.message.find("as far as the arm reaches"), std::string::npos)
        << result.message;
    EXPECT_LT(result.seconds, 30.0);
    EXPECT_LE(result.peak_tilt, 0.5 + 1e-9);
}

TEST(PourPlanner, AGranularSeekCrossesItsReposeAngleAtTheTiltRate)
{
    // Nothing granular reaches the lip until its surface has tipped past its
    // angle of repose, so that stretch is crossed at the tilt rate like the
    // rest of seek_fast_until.
    auto plain = make_tapping_planner(0.0, 0.35);
    auto repose = make_tapping_planner(0.5, 0.35);
    PourRequest sugar = water_request(20.0);
    sugar.material = MaterialClass::Granular;
    RunOptions opt;
    opt.sim_gain = 30.0;
    opt.sim_transport_delay = 0.3;
    opt.sim_tail = 0.5;
    opt.sim_onset = 0.95;
    const auto a = run_pour(plain, sugar, opt);
    const auto b = run_pour(repose, sugar, opt);
    ASSERT_TRUE(a.finished && b.finished);
    EXPECT_TRUE(b.success) << b.message;
    EXPECT_NEAR(b.delivered, 20.0, 1.5);
    // 0.35 -> 0.85 rad at salt's 0.35 rad/s instead of its 0.10.
    EXPECT_LT(b.seconds, a.seconds - 3.0) << a.seconds << " s plain, " << b.seconds << " s";
}
