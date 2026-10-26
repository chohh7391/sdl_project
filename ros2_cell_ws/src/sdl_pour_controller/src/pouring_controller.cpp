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

#include "sdl_pour_controller/pouring_controller.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <vector>

namespace sdl_pour_controller {

using pour::MaterialClass;
using pour::PourLimits;
using pour::PourPhase;

namespace {
// The IK throttle: the path runs at full speed while the residual is under
// kSlowFrom of the tolerance and slows to a halt at the tolerance. Within
// kBlockedRatio of it the throttle has all but stopped the path, and held
// there for ik_fail_sec the arm is not following -- it cannot.
constexpr double kSlowFrom = 0.5;
constexpr double kBlockedRatio = 0.95;
// A tap may use this much of the pour joint's per-cycle bound; the rest is
// left for the tilt it rides on.
constexpr double kTapRateShare = 0.8;
// How far short of the tilt bound a tap stops [rad].
constexpr double kTapMargin = 0.005;
// On the way back a block is waited out this many ik_fail_sec before the arm
// is sent home in joint space. The way back retraces the way in, so a real
// block there is unlikely, and a false one leaves the lip path beside the
// receiver.
constexpr double kReturnBlockFactor = 4.0;
// The share of the joints' per-cycle bound (max_delta_q) the tilt is planned
// to use, and of the lip's speed. The rest is headroom: the cap is taken from
// the path's derivative where the tilt is now, and the path bends.
constexpr double kIkRateShare = 0.7;
constexpr double kLipRateShare = 0.9;
// How much harder than its own acceleration a path brakes once the IK is
// ik_hold_ratio behind -- instead of stopping it in one cycle, which is a
// knock -- and how far behind it is stopped dead all the same, the last resort
// when the cap above has not kept it from falling behind.
constexpr double kEmergencyBrake = 4.0;
constexpr double kDeadHoldRatio = 2.0;
// The step the path's derivatives are taken over [rad of tilt].
constexpr double kPathEps = 1e-3;
// How far ahead of the tilt, in the direction it moves, the lip looks for
// where the path's inset limit closes in [rad], and in how many samples. The
// limit is a minimum over sampled points of the hand and the vessel, each of
// which only counts once it is below the rim, so it closes in steps: on the
// way back upright on the rig (2026-10-04 23:3x) by 0.3-0.6 mm every 1.5-2.6
// deg between 25 and 11 deg. Backed out only when the tilt got there, the lip
// held the tilt for a cycle or two each time and the tilt restarted from
// rest: five stop-and-go knocks in 3 s.
// Heading straight for the lowest limit in that stretch still sent the lip
// out at its full 30 mm/s for a cycle or two at every step -- its pace is 8 --
// five small jolts in a return (2026-10-04 23:4x). It heads instead for the
// lowest of the limit plus kLipEnvelopeSlope per radian still to go: each
// step is taken up over the stretch before it, at most 1.4 mm/deg, which at
// 0.15 rad/s is 12 mm/s. The stretch is long enough for kLipEnvelopeSlope to
// take up 4 mm, several times the steps seen, and sampled finely enough that
// the target slides rather than steps.
constexpr double kLipLookahead = 0.05;
constexpr double kLipEnvelopeSlope = 0.08;
// In toward the limit the lip goes no faster than the tilt carries it --
// kLipEnvelopeSlope per radian -- plus this creep [m/s], so it still gets
// there while the tilt holds. Tipping up the limit opens in the same steps it
// closes in on the way back, and the lip went in at its full 30 mm/s at each:
// 23 and 30 mm/s spikes from a 1.5 mm/s pace in the slow seek, 22-26 deg
// (2026-10-05 00:14). Out, away from the receiver, it keeps its full speed.
constexpr double kLipCreep = 0.002;
// The stretch is looked over cell by cell, and each cell's limit is computed
// once a pour and kept: the limit is a loop over every sampled point of the
// hand and the vessel (~2400 here), and a hundred of those a cycle would take
// milliseconds of the 8 the control loop has. The exact limit still guards
// every step the tilt actually takes.
constexpr double kLipTableStep = 0.0005;

// Seeds, not measurements, except where noted. The liquid/free endpoint is the
// one with hardware behind it: water on an HS-AA, 2026-09-16, which ran 6-11 g/s
// continuous, settled 1.0 s after the flow stopped, left <= 0.21 g of tail, and
// broke into ~1.2 g drops once the stream thinned. Everything else is a
// commissioning starting point and belongs in sdl_cell_bringup's
// config/{mujoco,real}/pouring_controller.yaml, not here.
PourLimits liquid_free_defaults()
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

PourLimits liquid_resistant_defaults()
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
    // Not 0.5 g. Every stop of a material this thick leaves a string behind, and
    // that string is the smallest amount it can be metered in: asking for finer
    // than the tail spends the whole pulse budget oscillating around a target
    // the material cannot hit.
    l.dose_quantum = 3.0;
    return l;
}

PourLimits granular_free_defaults()
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
    l.repose_angle = 0.5;
    l.shake_amplitude = 0.02;
    l.shake_period = 0.25;
    return l;
}

PourLimits granular_resistant_defaults()
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
    l.repose_angle = 0.6;
    l.shake_amplitude = 0.03;
    l.shake_period = 0.2;
    return l;
}

Eigen::Isometry3d to_isometry(const pinocchio::SE3 & H)
{
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() = H.rotation();
    T.translation() = H.translation();
    return T;
}
}  // namespace

controller_interface::InterfaceConfiguration
PouringController::command_interface_configuration() const
{
    controller_interface::InterfaceConfiguration config;
    config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
    for (const auto & name : joint_names_) {
        config.names.push_back(name + "/position");
    }
    return config;
}

void PouringController::declare_limits(const std::string & prefix, const PourLimits & d)
{
    auto_declare<double>(prefix + ".max_flow_rate", d.max_flow_rate);
    auto_declare<double>(prefix + ".trim_flow_rate", d.trim_flow_rate);
    auto_declare<double>(prefix + ".trim_pulse_sec", d.trim_pulse_sec);
    auto_declare<double>(prefix + ".transport_delay", d.transport_delay);
    auto_declare<double>(prefix + ".afterflow_grams", d.afterflow_grams);
    auto_declare<double>(prefix + ".settle_hold_sec", d.settle_hold_sec);
    auto_declare<double>(prefix + ".tilt_rate", d.tilt_rate);
    auto_declare<double>(prefix + ".seek_tilt_rate", d.seek_tilt_rate);
    auto_declare<double>(prefix + ".dose_quantum", d.dose_quantum);
    auto_declare<double>(prefix + ".park_drop", d.park_drop);
    auto_declare<double>(prefix + ".forecast_min_pairs", d.forecast_min_pairs);
    auto_declare<double>(prefix + ".slow_seek_below_grams", d.slow_seek_below_grams);
    auto_declare<double>(prefix + ".early_flight_factor", d.early_flight_factor);
    auto_declare<double>(prefix + ".keep_flowing_margin", d.keep_flowing_margin);
    auto_declare<double>(prefix + ".stop_forecast_scale", d.stop_forecast_scale);
    auto_declare<double>(prefix + ".max_trim_pulses", d.max_trim_pulses);
    auto_declare<double>(prefix + ".final_approach_grams", d.final_approach_grams);
    auto_declare<double>(prefix + ".final_approach_rise", d.final_approach_rise);
    auto_declare<double>(prefix + ".max_step_down", d.max_step_down);
    auto_declare<double>(prefix + ".final_approach_shake_amplitude", d.final_approach_shake_amplitude);
    auto_declare<double>(prefix + ".final_approach_shake_period", d.final_approach_shake_period);
    auto_declare<double>(prefix + ".final_approach_tilt_rate", d.final_approach_tilt_rate);
    auto_declare<double>(prefix + ".repose_angle", d.repose_angle);
    auto_declare<double>(prefix + ".shake_amplitude", d.shake_amplitude);
    auto_declare<double>(prefix + ".shake_period", d.shake_period);
}

PourLimits PouringController::read_limits(const std::string & prefix) const
{
    auto node = get_node();
    PourLimits l;
    l.max_flow_rate = node->get_parameter(prefix + ".max_flow_rate").as_double();
    l.trim_flow_rate = node->get_parameter(prefix + ".trim_flow_rate").as_double();
    l.trim_pulse_sec = node->get_parameter(prefix + ".trim_pulse_sec").as_double();
    l.transport_delay = node->get_parameter(prefix + ".transport_delay").as_double();
    l.afterflow_grams = node->get_parameter(prefix + ".afterflow_grams").as_double();
    l.settle_hold_sec = node->get_parameter(prefix + ".settle_hold_sec").as_double();
    l.tilt_rate = node->get_parameter(prefix + ".tilt_rate").as_double();
    l.seek_tilt_rate = node->get_parameter(prefix + ".seek_tilt_rate").as_double();
    l.dose_quantum = node->get_parameter(prefix + ".dose_quantum").as_double();
    l.park_drop = node->get_parameter(prefix + ".park_drop").as_double();
    l.forecast_min_pairs = node->get_parameter(prefix + ".forecast_min_pairs").as_double();
    l.slow_seek_below_grams = node->get_parameter(prefix + ".slow_seek_below_grams").as_double();
    l.early_flight_factor = node->get_parameter(prefix + ".early_flight_factor").as_double();
    l.keep_flowing_margin = node->get_parameter(prefix + ".keep_flowing_margin").as_double();
    l.stop_forecast_scale = node->get_parameter(prefix + ".stop_forecast_scale").as_double();
    l.max_trim_pulses = node->get_parameter(prefix + ".max_trim_pulses").as_double();
    l.final_approach_grams = node->get_parameter(prefix + ".final_approach_grams").as_double();
    l.final_approach_rise = node->get_parameter(prefix + ".final_approach_rise").as_double();
    l.max_step_down = node->get_parameter(prefix + ".max_step_down").as_double();
    l.final_approach_shake_amplitude =
        node->get_parameter(prefix + ".final_approach_shake_amplitude").as_double();
    l.final_approach_shake_period =
        node->get_parameter(prefix + ".final_approach_shake_period").as_double();
    l.final_approach_tilt_rate = node->get_parameter(prefix + ".final_approach_tilt_rate").as_double();
    l.repose_angle = node->get_parameter(prefix + ".repose_angle").as_double();
    l.shake_amplitude = node->get_parameter(prefix + ".shake_amplitude").as_double();
    l.shake_period = node->get_parameter(prefix + ".shake_period").as_double();
    return l;
}

bool PouringController::read_profile(const std::string & prefix, pour::MaterialProfile & out,
                                     std::string & why)
{
    out.free = read_limits(prefix + ".free");
    out.resistant = read_limits(prefix + ".resistant");
    return out.validate(prefix, why);
}

CallbackReturn PouringController::on_init()
{
    if (FR5BaseController::on_init() != CallbackReturn::SUCCESS) {
        return CallbackReturn::FAILURE;
    }
    try {
        auto_declare<std::string>("scale_topic", scale_topic_);
        auto_declare<std::string>("pour_joint", "");
        auto_declare<double>("pour_direction", pour_direction_);
        auto_declare<std::string>("granular.tilt_about", granular_tilt_about_);
        auto_declare<std::string>("liquid.tilt_about", liquid_tilt_about_);
        auto_declare<double>("granular.pour_direction", granular_pour_direction_);
        auto_declare<double>("granular.lip_height", granular_lip_height_);
        auto_declare<double>("granular.lip_inset", granular_lip_inset_);
        auto_declare<double>("granular.max_tilt", granular_max_tilt_);
        auto_declare<double>("granular.timeout", granular_timeout_);
        auto_declare<double>("scale_timeout", scale_timeout_);
        auto_declare<double>("scale_sample_period", scale_sample_period_);
        auto_declare<double>("outlier_factor", outlier_factor_);
        auto_declare<double>("max_tilt", max_tilt_);
        auto_declare<double>("max_back_tilt", max_back_tilt_);
        auto_declare<double>("weight_tolerance", weight_tolerance_);
        auto_declare<double>("max_delta_q", max_delta_q_);
        auto_declare<double>("tilt_accel", tilt_accel_);
        auto_declare<double>("tilt_jerk", tilt_jerk_);
        auto_declare<double>("ik_hold_ratio", ik_hold_ratio_);
        auto_declare<double>("feedback_period", feedback_period_);

        const pour::PlannerConfig d;
        auto_declare<double>("container_tolerance", d.container_tolerance);
        auto_declare<double>("verify_timeout", d.verify_timeout);
        auto_declare<double>("onset_grams", d.onset_grams);
        auto_declare<double>("approach_sec", d.approach_sec);
        auto_declare<double>("kp_tilt", d.kp_tilt);
        auto_declare<double>("max_tilt_lead", d.max_tilt_lead);
        auto_declare<double>("max_tilt_lead_far", d.max_tilt_lead_far);
        auto_declare<double>("lead_fraction", d.lead_fraction);
        auto_declare<double>("seek_fast_until", d.seek_fast_until);
        auto_declare<double>("slow_seek_min_fraction", d.slow_seek_min_fraction);
        auto_declare<double>("flow_deadband", d.flow_deadband);
        auto_declare<double>("no_flow_epsilon", d.no_flow_epsilon);
        auto_declare<double>("park_check_sec", d.park_check_sec);
        auto_declare<double>("park_drip_grams", d.park_drip_grams);
        auto_declare<double>("stop_margin_factor", d.stop_margin_factor);
        auto_declare<double>("granular.stop_margin_factor", d.granular_stop_margin_factor);
        auto_declare<double>("flow_model.prior_gain", d.flow_model.prior_gain);
        auto_declare<double>("flow_model.prior_gain_sd", d.flow_model.prior_gain_sd);
        auto_declare<double>("flow_model.prior_onset_sd", d.flow_model.prior_onset_sd);
        auto_declare<double>("flow_model.prior_rise", d.flow_model.prior_rise);
        auto_declare<double>("flow_model.prior_rise_sd", d.flow_model.prior_rise_sd);
        auto_declare<double>("flow_model.flow_noise", d.flow_model.flow_noise);
        auto_declare<int>("flow_model.min_pairs", d.flow_model.min_pairs);
        auto_declare<double>("flow_model.max_onset_shift", d.flow_model.max_onset_shift);
        auto_declare<double>("settle_timeout", d.settle_timeout);
        auto_declare<double>("stall_timeout", d.stall_timeout);
        auto_declare<double>("trim_undershoot", d.trim_undershoot);
        auto_declare<int>("max_trim_pulses", d.max_trim_pulses);
        auto_declare<double>("retract_margin", d.retract_margin);
        auto_declare<double>("trim_detect_grams", d.trim_detect_grams);
        auto_declare<double>("trim_creep_fraction", d.trim_creep_fraction);
        auto_declare<double>("tilt_epsilon", d.tilt_epsilon);
        auto_declare<int>("max_park_attempts", d.max_park_attempts);
        auto_declare<double>("retract_slack_sec", d.retract_slack_sec);
        auto_declare<double>("max_pulse_sec", d.max_pulse_sec);

        auto_declare<std::string>("law", law_name_);
        const pour::ShapingConfig sd;
        auto_declare<double>("shaping.control_period", sd.control_period);
        auto_declare<double>("shaping.kp", sd.kp);
        auto_declare<double>("shaping.kd", sd.kd);
        auto_declare<double>("shaping.kernel_horizon", sd.kernel_horizon);
        auto_declare<double>("shaping.shaping_freq", sd.shaping_freq);
        auto_declare<double>("shaping.shaping_decay", sd.shaping_decay);
        auto_declare<double>("shaping.kernel_alpha", sd.kernel_alpha);
        auto_declare<double>("shaping.max_tilt_rate", sd.max_tilt_rate);
        auto_declare<double>("shaping.settle_hold_sec", sd.settle_hold_sec);

        // ---- where the lip goes: `joint` or `measured` ----
        auto_declare<std::string>("pour_geometry", "joint");
        auto_declare<std::string>("geometry.vessel_pose_topic", "");
        auto_declare<std::string>("geometry.vessel_pose_frame", vessel_pose_frame_);
        auto_declare<double>("geometry.vessel_settle_sec", vessel_settle_sec_);
        auto_declare<double>("geometry.vessel_pose_max_age", vessel_pose_max_age_);
        auto_declare<double>("geometry.vessel_pose_timeout", vessel_pose_timeout_);
        auto_declare<double>("geometry.vessel_radius", vessel_.radius);
        auto_declare<double>("geometry.vessel_height", vessel_.height);
        // No defaults that would pass for measurements: where the marker is
        // stuck and where the receiver stands are facts about this bench. NaN,
        // not zero, because zero is a real answer (see configure).
        auto_declare<double>("geometry.tag_radius", std::numeric_limits<double>::quiet_NaN());
        auto_declare<double>("geometry.tag_height", std::numeric_limits<double>::quiet_NaN());
        auto_declare<std::vector<double>>("geometry.receiver_rim_center", {});
        auto_declare<double>("geometry.receiver_radius", receiver_.radius);
        auto_declare<double>("geometry.receiver_rim_z", receiver_rim_z_);
        const pour::LipPathConfig ld;
        auto_declare<double>("geometry.lip_clearance", ld.clearance);
        auto_declare<double>("geometry.lip_max_height", ld.max_height);
        auto_declare<double>("geometry.lip_height", ld.lip_height);
        auto_declare<double>("geometry.lip_inset", ld.inset);
        auto_declare<double>("geometry.lip_gap", ld.gap);
        auto_declare<double>("geometry.landing_margin", ld.landing_margin);
        auto_declare<double>("geometry.landing_by_tilt", ld.landing_by_tilt);
        auto_declare<double>("geometry.max_align_distance", ld.max_align_distance);
        auto_declare<double>("geometry.max_axis_elevation", ld.max_axis_elevation);
        auto_declare<double>("geometry.max_centering_error", ld.max_centering_error);
        auto_declare<double>("geometry.max_tag_distance", ld.max_tag_distance);
        auto_declare<std::vector<double>>("geometry.closing_axis_in_ee", {1.0, 0.0, 0.0});
        auto_declare<std::vector<double>>("geometry.approach_axis_in_ee", {0.0, 0.0, 1.0});
        auto_declare<double>("geometry.grasp_depth", ld.grasp_depth);
        auto_declare<double>("geometry.max_depth_error", ld.max_depth_error);
        auto_declare<double>("geometry.max_carry_lean", max_carry_lean_);
        // Standing the vessel upright before the pour (pour/level.hpp).
        const pour::LevelConfig lv;
        auto_declare<bool>("geometry.level.enabled", level_enabled_);
        auto_declare<std::string>("geometry.level.attitude_topic", "");
        auto_declare<bool>("geometry.level.require_tag", level_require_tag_);
        auto_declare<double>("geometry.level.rate", level_rate_);
        auto_declare<double>("geometry.level.max_correction", lv.max_correction);
        auto_declare<double>("geometry.level.max_tag_disagreement", lv.max_tag_disagreement);
        auto_declare<double>("geometry.level.deadband", lv.deadband);
        auto_declare<std::vector<double>>("geometry.hand_boxes", {});
        auto_declare<double>("geometry.align_speed", align_speed_);
        auto_declare<double>("geometry.align_accel", align_accel_);
        auto_declare<double>("geometry.max_lip_speed", max_lip_speed_);
        auto_declare<double>("geometry.ik_lambda", ik_lambda_);
        auto_declare<double>("geometry.ik_tolerance", ik_tolerance_);
        auto_declare<double>("geometry.ik_rot_tolerance", ik_rot_tolerance_);
        auto_declare<double>("geometry.ik_fail_sec", ik_fail_sec_);

        declare_limits("liquid.free", liquid_free_defaults());
        declare_limits("liquid.resistant", liquid_resistant_defaults());
        declare_limits("granular.free", granular_free_defaults());
        declare_limits("granular.resistant", granular_resistant_defaults());
    } catch (const std::exception & e) {
        RCLCPP_ERROR(get_node()->get_logger(), "Init exception: %s", e.what());
        return CallbackReturn::ERROR;
    }
    return CallbackReturn::SUCCESS;
}

bool PouringController::assign_parameters()
{
    auto node = get_node();
    auto logger = node->get_logger();

    scale_topic_ = node->get_parameter("scale_topic").as_string();
    pour_joint_ = node->get_parameter("pour_joint").as_string();
    pour_direction_ = node->get_parameter("pour_direction").as_double();
    granular_tilt_about_ = node->get_parameter("granular.tilt_about").as_string();
    liquid_tilt_about_ = node->get_parameter("liquid.tilt_about").as_string();
    granular_pour_direction_ = node->get_parameter("granular.pour_direction").as_double();
    granular_lip_height_ = node->get_parameter("granular.lip_height").as_double();
    granular_lip_inset_ = node->get_parameter("granular.lip_inset").as_double();
    granular_max_tilt_ = node->get_parameter("granular.max_tilt").as_double();
    granular_timeout_ = node->get_parameter("granular.timeout").as_double();
    scale_timeout_ = node->get_parameter("scale_timeout").as_double();
    scale_sample_period_ = node->get_parameter("scale_sample_period").as_double();
    outlier_factor_ = node->get_parameter("outlier_factor").as_double();
    max_tilt_ = node->get_parameter("max_tilt").as_double();
    max_back_tilt_ = node->get_parameter("max_back_tilt").as_double();
    weight_tolerance_ = node->get_parameter("weight_tolerance").as_double();
    max_delta_q_ = node->get_parameter("max_delta_q").as_double();
    tilt_accel_ = node->get_parameter("tilt_accel").as_double();
    tilt_jerk_ = node->get_parameter("tilt_jerk").as_double();
    ik_hold_ratio_ = node->get_parameter("ik_hold_ratio").as_double();
    feedback_period_ = node->get_parameter("feedback_period").as_double();
    if (!std::isfinite(tilt_jerk_) || tilt_jerk_ < 0.0) {
        RCLCPP_ERROR(logger, "tilt_jerk must be >= 0 (0: no jerk bound), got %f", tilt_jerk_);
        return false;
    }
    if (!std::isfinite(ik_hold_ratio_) || ik_hold_ratio_ < 1.0 || ik_hold_ratio_ >= 2.0) {
        RCLCPP_ERROR(logger,
            "ik_hold_ratio must be in [1, 2): the path is braked hard once the IK is that many "
            "tolerances behind, and stopped dead at 2, beyond which it would run on well outside "
            "the lip's gap; got %f",
            ik_hold_ratio_);
        return false;
    }
    tilt_ramp_.set_jerk(tilt_jerk_);

    planner_config_.container_tolerance = node->get_parameter("container_tolerance").as_double();
    planner_config_.verify_timeout = node->get_parameter("verify_timeout").as_double();
    planner_config_.onset_grams = node->get_parameter("onset_grams").as_double();
    planner_config_.approach_sec = node->get_parameter("approach_sec").as_double();
    planner_config_.kp_tilt = node->get_parameter("kp_tilt").as_double();
    planner_config_.max_tilt_lead = node->get_parameter("max_tilt_lead").as_double();
    planner_config_.max_tilt_lead_far = node->get_parameter("max_tilt_lead_far").as_double();
    planner_config_.lead_fraction = node->get_parameter("lead_fraction").as_double();
    planner_config_.seek_fast_until = node->get_parameter("seek_fast_until").as_double();
    planner_config_.slow_seek_min_fraction = node->get_parameter("slow_seek_min_fraction").as_double();
    planner_config_.flow_deadband = node->get_parameter("flow_deadband").as_double();
    planner_config_.no_flow_epsilon = node->get_parameter("no_flow_epsilon").as_double();
    planner_config_.park_check_sec = node->get_parameter("park_check_sec").as_double();
    planner_config_.park_drip_grams = node->get_parameter("park_drip_grams").as_double();
    planner_config_.stop_margin_factor = node->get_parameter("stop_margin_factor").as_double();
    planner_config_.granular_stop_margin_factor =
        node->get_parameter("granular.stop_margin_factor").as_double();
    {
        auto & fm = planner_config_.flow_model;
        fm.prior_gain = node->get_parameter("flow_model.prior_gain").as_double();
        fm.prior_gain_sd = node->get_parameter("flow_model.prior_gain_sd").as_double();
        fm.prior_onset_sd = node->get_parameter("flow_model.prior_onset_sd").as_double();
        fm.prior_rise = node->get_parameter("flow_model.prior_rise").as_double();
        fm.prior_rise_sd = node->get_parameter("flow_model.prior_rise_sd").as_double();
        fm.flow_noise = node->get_parameter("flow_model.flow_noise").as_double();
        fm.min_pairs = static_cast<int>(node->get_parameter("flow_model.min_pairs").as_int());
        fm.max_onset_shift = node->get_parameter("flow_model.max_onset_shift").as_double();
    }
    planner_config_.settle_timeout = node->get_parameter("settle_timeout").as_double();
    planner_config_.stall_timeout = node->get_parameter("stall_timeout").as_double();
    planner_config_.trim_undershoot = node->get_parameter("trim_undershoot").as_double();
    planner_config_.max_trim_pulses = static_cast<int>(node->get_parameter("max_trim_pulses").as_int());
    planner_config_.retract_margin = node->get_parameter("retract_margin").as_double();
    planner_config_.trim_detect_grams = node->get_parameter("trim_detect_grams").as_double();
    planner_config_.trim_creep_fraction = node->get_parameter("trim_creep_fraction").as_double();
    planner_config_.tilt_epsilon = node->get_parameter("tilt_epsilon").as_double();
    planner_config_.max_park_attempts = static_cast<int>(node->get_parameter("max_park_attempts").as_int());
    planner_config_.retract_slack_sec = node->get_parameter("retract_slack_sec").as_double();
    planner_config_.max_pulse_sec = node->get_parameter("max_pulse_sec").as_double();

    // Every one of these bounds a real motion, so a nonsense value is refused at
    // configure rather than discovered mid-pour with a vessel in the gripper.
    const std::pair<const char *, double> positives[] = {
        {"scale_timeout", scale_timeout_},
        {"scale_sample_period", scale_sample_period_},
        {"outlier_factor", outlier_factor_},
        {"max_tilt", max_tilt_},
        {"weight_tolerance", weight_tolerance_},
        {"max_delta_q", max_delta_q_},
        {"tilt_accel", tilt_accel_},
        {"feedback_period", feedback_period_},
    };
    for (const auto & [name, value] : positives) {
        if (!std::isfinite(value) || value <= 0.0) {
            RCLCPP_ERROR(logger, "%s must be positive (got %f)", name, value);
            return false;
        }
    }
    // Zero is legal -- the vessel may never be turned behind the attitude it
    // was carried in -- but a negative or non-finite bound is not a bound.
    if (!std::isfinite(max_back_tilt_) || max_back_tilt_ < 0.0) {
        RCLCPP_ERROR(logger, "max_back_tilt must be zero or positive (got %f)", max_back_tilt_);
        return false;
    }
    if (scale_topic_.empty()) {
        RCLCPP_ERROR(logger, "scale_topic must name the topic the weight arrives on");
        return false;
    }
    // Not "anything nonzero": the sign is applied to a joint command, and a
    // pour_direction of 0.7 would silently scale every tilt the planner asked
    // for by 0.7 and make every rate and bound in pour/ a lie.
    if (pour_direction_ != 1.0 && pour_direction_ != -1.0) {
        RCLCPP_ERROR(logger,
            "pour_direction must be exactly 1.0 or -1.0 (got %f): it says which way the pour "
            "joint turns to bring the lip down, and guessing it wrong tips the vessel away "
            "from the scale", pour_direction_);
        return false;
    }
    if (liquid_tilt_about_ != "goal" && liquid_tilt_about_ != "fingertips") {
        RCLCPP_ERROR(logger,
            "liquid.tilt_about must be \"goal\" or \"fingertips\" (got \"%s\")",
            liquid_tilt_about_.c_str());
        return false;
    }
    if (granular_tilt_about_ != "goal" && granular_tilt_about_ != "pour_joint") {
        RCLCPP_ERROR(logger,
            "granular.tilt_about must be \"goal\" or \"pour_joint\" (got \"%s\")",
            granular_tilt_about_.c_str());
        return false;
    }
    if (!std::isfinite(granular_lip_height_) || granular_lip_height_ < 0.0 ||
        granular_lip_height_ > 0.2) {
        RCLCPP_ERROR(logger,
            "granular.lip_height must be 0 (use geometry.lip_height) or up to 0.2 m (got %f)",
            granular_lip_height_);
        return false;
    }
    if (!std::isfinite(granular_lip_inset_) || granular_lip_inset_ < 0.0 ||
        granular_lip_inset_ > 0.1) {
        RCLCPP_ERROR(logger,
            "granular.lip_inset must be 0 (use geometry.lip_inset) or up to 0.1 m (got %f)",
            granular_lip_inset_);
        return false;
    }
    if (!std::isfinite(granular_max_tilt_) || granular_max_tilt_ < 0.0 ||
        granular_max_tilt_ > M_PI) {
        RCLCPP_ERROR(logger,
            "granular.max_tilt must be 0 (use max_tilt) or up to pi rad (got %f)",
            granular_max_tilt_);
        return false;
    }
    if (!std::isfinite(granular_timeout_) || granular_timeout_ < 0.0) {
        RCLCPP_ERROR(logger,
            "granular.timeout must be 0 (use the liquid's) or positive seconds (got %f)",
            granular_timeout_);
        return false;
    }
    if (granular_pour_direction_ != 0.0 && granular_pour_direction_ != 1.0 &&
        granular_pour_direction_ != -1.0) {
        RCLCPP_ERROR(logger,
            "granular.pour_direction must be 0 (use pour_direction), 1.0 or -1.0 (got %f)",
            granular_pour_direction_);
        return false;
    }
    // The scale's own period has to be shorter than the staleness bound, or a
    // perfectly healthy stream reads as a dead one on its very first sample.
    if (scale_timeout_ <= scale_sample_period_) {
        RCLCPP_ERROR(logger,
            "scale_timeout (%f s) must exceed scale_sample_period (%f s); a healthy stream "
            "would otherwise be declared stale between samples",
            scale_timeout_, scale_sample_period_);
        return false;
    }

    std::string why;
    if (!read_profile("liquid", liquid_, why) || !read_profile("granular", granular_, why)) {
        RCLCPP_ERROR(logger, "Material profile rejected: %s", why.c_str());
        return false;
    }
    if (!planner_.configure(planner_config_, liquid_, granular_, why)) {
        RCLCPP_ERROR(logger, "Pour planner rejected its configuration: %s", why.c_str());
        return false;
    }

    // The shaping law's own gains, plus the obligations it shares with the
    // phase machine taken from the SAME parameters -- two laws being compared
    // must be judged by one container check, one settle and one stall rule.
    shaping_config_.control_period = node->get_parameter("shaping.control_period").as_double();
    shaping_config_.kp = node->get_parameter("shaping.kp").as_double();
    shaping_config_.kd = node->get_parameter("shaping.kd").as_double();
    shaping_config_.kernel_horizon = node->get_parameter("shaping.kernel_horizon").as_double();
    shaping_config_.shaping_freq = node->get_parameter("shaping.shaping_freq").as_double();
    shaping_config_.shaping_decay = node->get_parameter("shaping.shaping_decay").as_double();
    shaping_config_.kernel_alpha = node->get_parameter("shaping.kernel_alpha").as_double();
    shaping_config_.max_tilt_rate = node->get_parameter("shaping.max_tilt_rate").as_double();
    shaping_config_.settle_hold_sec = node->get_parameter("shaping.settle_hold_sec").as_double();
    shaping_config_.container_tolerance = planner_config_.container_tolerance;
    shaping_config_.verify_timeout = planner_config_.verify_timeout;
    shaping_config_.settle_timeout = planner_config_.settle_timeout;
    shaping_config_.stall_timeout = planner_config_.stall_timeout;
    shaping_config_.no_flow_epsilon = planner_config_.no_flow_epsilon;
    shaping_config_.tilt_epsilon = planner_config_.tilt_epsilon;
    if (!shaping_.configure(shaping_config_, why)) {
        RCLCPP_ERROR(logger, "Shaping pour law rejected its configuration: %s", why.c_str());
        return false;
    }

    law_name_ = node->get_parameter("law").as_string();
    if (law_name_ == "phase_machine") {
        law_ = &planner_;
    } else if (law_name_ == "shaping") {
        law_ = &shaping_;
    } else {
        RCLCPP_ERROR(logger,
            "law must be 'phase_machine' or 'shaping' (got '%s'). They are two different "
            "answers to one question -- how to pour against a delayed scale -- and which is "
            "better on this cell is what running both is for", law_name_.c_str());
        return false;
    }

    const std::string geometry = node->get_parameter("pour_geometry").as_string();
    if (geometry == "joint") {
        measured_geometry_ = false;
    } else if (geometry == "measured") {
        measured_geometry_ = true;
        if (!assign_geometry_parameters()) {
            return false;
        }
    } else {
        RCLCPP_ERROR(logger,
            "pour_geometry must be 'joint' or 'measured' (got '%s'): turn the pour joint alone, "
            "or measure where the vessel sits in the gripper and tip it about its lip",
            geometry.c_str());
        return false;
    }
    return true;
}

bool PouringController::assign_geometry_parameters()
{
    auto node = get_node();
    auto logger = node->get_logger();

    const auto vec3 = [&](const char * name, const char * meaning, Eigen::Vector3d & out) {
        const auto v = node->get_parameter(name).as_double_array();
        if (v.size() != 3 || !std::all_of(v.begin(), v.end(), [](double x) { return std::isfinite(x); })) {
            RCLCPP_ERROR(logger, "%s must be three finite numbers: %s", name, meaning);
            return false;
        }
        out = Eigen::Vector3d(v[0], v[1], v[2]);
        return true;
    };

    vessel_pose_topic_ = node->get_parameter("geometry.vessel_pose_topic").as_string();
    vessel_pose_frame_ = node->get_parameter("geometry.vessel_pose_frame").as_string();
    vessel_settle_sec_ = node->get_parameter("geometry.vessel_settle_sec").as_double();
    vessel_pose_max_age_ = node->get_parameter("geometry.vessel_pose_max_age").as_double();
    vessel_pose_timeout_ = node->get_parameter("geometry.vessel_pose_timeout").as_double();
    vessel_.radius = node->get_parameter("geometry.vessel_radius").as_double();
    vessel_.height = node->get_parameter("geometry.vessel_height").as_double();
    receiver_.radius = node->get_parameter("geometry.receiver_radius").as_double();
    receiver_rim_z_ = node->get_parameter("geometry.receiver_rim_z").as_double();
    lip_config_.clearance = node->get_parameter("geometry.lip_clearance").as_double();
    lip_config_.max_height = node->get_parameter("geometry.lip_max_height").as_double();
    lip_config_.lip_height = node->get_parameter("geometry.lip_height").as_double();
    lip_config_.inset = node->get_parameter("geometry.lip_inset").as_double();
    lip_config_.gap = node->get_parameter("geometry.lip_gap").as_double();
    lip_config_.landing_margin = node->get_parameter("geometry.landing_margin").as_double();
    lip_config_.landing_by_tilt = node->get_parameter("geometry.landing_by_tilt").as_double();
    lip_config_.max_align_distance = node->get_parameter("geometry.max_align_distance").as_double();
    lip_config_.max_axis_elevation = node->get_parameter("geometry.max_axis_elevation").as_double();
    lip_config_.max_centering_error = node->get_parameter("geometry.max_centering_error").as_double();
    lip_config_.max_tag_distance = node->get_parameter("geometry.max_tag_distance").as_double();
    align_speed_ = node->get_parameter("geometry.align_speed").as_double();
    align_accel_ = node->get_parameter("geometry.align_accel").as_double();
    max_lip_speed_ = node->get_parameter("geometry.max_lip_speed").as_double();
    ik_lambda_ = node->get_parameter("geometry.ik_lambda").as_double();
    ik_tolerance_ = node->get_parameter("geometry.ik_tolerance").as_double();
    ik_rot_tolerance_ = node->get_parameter("geometry.ik_rot_tolerance").as_double();
    ik_fail_sec_ = node->get_parameter("geometry.ik_fail_sec").as_double();

    if (vessel_pose_topic_.empty() || vessel_pose_frame_.empty()) {
        RCLCPP_ERROR(logger,
            "geometry.vessel_pose_topic and geometry.vessel_pose_frame must be set under "
            "pour_geometry: measured -- the PoseStamped of the held vessel's marker, and the "
            "frame it has to be in (the arm's base; nothing here transforms it)");
        return false;
    }
    vessel_.tag_radius = node->get_parameter("geometry.tag_radius").as_double();
    vessel_.tag_height = node->get_parameter("geometry.tag_height").as_double();
    lip_config_.grasp_depth = node->get_parameter("geometry.grasp_depth").as_double();
    lip_config_.max_depth_error = node->get_parameter("geometry.max_depth_error").as_double();
    max_carry_lean_ = node->get_parameter("geometry.max_carry_lean").as_double();
    // Unset is NaN. Zero is an answer: the pose already IS a point on the
    // vessel's axis, because the object table applied a known offset in the
    // marker's own yaw -- which, unlike a bare radius, fixes the depth along
    // the jaws even for a marker standing off to one side of them.
    if (!std::isfinite(vessel_.tag_radius) || vessel_.tag_radius < 0.0 ||
        !std::isfinite(vessel_.tag_height)) {
        RCLCPP_ERROR(logger,
            "geometry.tag_radius and geometry.tag_height must be measured: how far the "
            "pose's point sits from the held vessel's axis (0 when the object table puts it "
            "on the axis), and how high above its bottom (got %f, %f). Which side of the "
            "vessel it is on is not needed",
            vessel_.tag_radius, vessel_.tag_height);
        return false;
    }
    if (!std::isfinite(max_carry_lean_) || max_carry_lean_ <= 0.0) {
        RCLCPP_ERROR(logger, "geometry.max_carry_lean must be positive (got %f)", max_carry_lean_);
        return false;
    }
    level_enabled_ = node->get_parameter("geometry.level.enabled").as_bool();
    attitude_topic_ = node->get_parameter("geometry.level.attitude_topic").as_string();
    level_require_tag_ = node->get_parameter("geometry.level.require_tag").as_bool();
    level_rate_ = node->get_parameter("geometry.level.rate").as_double();
    level_config_.max_correction = node->get_parameter("geometry.level.max_correction").as_double();
    level_config_.max_tag_disagreement =
        node->get_parameter("geometry.level.max_tag_disagreement").as_double();
    level_config_.deadband = node->get_parameter("geometry.level.deadband").as_double();
    if (level_enabled_) {
        std::string why;
        if (!level_config_.validate(why)) {
            RCLCPP_ERROR(logger, "geometry.%s", why.c_str());
            return false;
        }
        if (!std::isfinite(level_rate_) || level_rate_ <= 0.0 || level_rate_ > 0.5) {
            RCLCPP_ERROR(logger,
                "geometry.level.rate must be in (0, 0.5] rad/s (got %f): the vessel is turned "
                "with whatever it holds in it", level_rate_);
            return false;
        }
        if (level_require_tag_ && attitude_topic_.empty()) {
            RCLCPP_ERROR(logger,
                "geometry.level.require_tag needs geometry.level.attitude_topic: there is no "
                "tag to require");
            return false;
        }
        // The vessel is turned about the point of its axis the jaws hold, and
        // that is found from the pose's point being on the axis at the bottom.
        if (vessel_.tag_radius != 0.0 || vessel_.tag_height != 0.0) {
            RCLCPP_ERROR(logger,
                "geometry.level needs the pose's point at the vessel's bottom centre "
                "(tag_radius 0, tag_height 0; got %f, %f): it turns the vessel about the "
                "point of its axis the jaws hold, found from there",
                vessel_.tag_radius, vessel_.tag_height);
            return false;
        }
    }
    if (!std::isfinite(receiver_rim_z_) || receiver_rim_z_ < 0.0 || receiver_rim_z_ > 1.0) {
        RCLCPP_ERROR(logger,
            "geometry.receiver_rim_z must be 0 (take the goal's) or a height in the base "
            "frame up to 1 m (got %f)", receiver_rim_z_);
        return false;
    }
    // Where the receiver stands comes with each goal, as perception measured
    // it. A position left here from when it was configured would be ignored
    // while looking authoritative, so it is refused instead.
    if (!node->get_parameter("geometry.receiver_rim_center").as_double_array().empty()) {
        RCLCPP_ERROR(logger,
            "geometry.receiver_rim_center is no longer read: the receiver is not fixed, and "
            "each pour goal says where its rim is (Pour.action receiver_rim_center). Remove "
            "it from the configuration");
        return false;
    }
    if (!vec3("geometry.approach_axis_in_ee",
              "the direction the jaws reach along, in the EE frame", lip_config_.approach_axis_in_ee) ||
        !vec3("geometry.closing_axis_in_ee", "the direction the jaws close along, in the EE frame",
              lip_config_.closing_axis_in_ee)) {
        return false;
    }
    const auto boxes = node->get_parameter("geometry.hand_boxes").as_double_array();
    if (boxes.size() % 6 != 0) {
        RCLCPP_ERROR(logger,
            "geometry.hand_boxes must be a flat list of six numbers per box (min x y z, max x y "
            "z, EE frame); got %zu numbers", boxes.size());
        return false;
    }
    lip_config_.hand_boxes.clear();
    for (std::size_t i = 0; i < boxes.size(); i += 6) {
        pour::HandBox box;
        box.min = Eigen::Vector3d(boxes[i], boxes[i + 1], boxes[i + 2]);
        box.max = Eigen::Vector3d(boxes[i + 3], boxes[i + 4], boxes[i + 5]);
        lip_config_.hand_boxes.push_back(box);
    }

    const std::pair<const char *, double> positives[] = {
        {"geometry.vessel_settle_sec", vessel_settle_sec_},
        {"geometry.vessel_pose_max_age", vessel_pose_max_age_},
        {"geometry.vessel_pose_timeout", vessel_pose_timeout_},
        {"geometry.vessel_radius", vessel_.radius},
        {"geometry.vessel_height", vessel_.height},
        {"geometry.receiver_radius", receiver_.radius},
        {"geometry.align_speed", align_speed_},
        {"geometry.align_accel", align_accel_},
        {"geometry.max_lip_speed", max_lip_speed_},
        {"geometry.ik_lambda", ik_lambda_},
        {"geometry.ik_tolerance", ik_tolerance_},
        {"geometry.ik_rot_tolerance", ik_rot_tolerance_},
        {"geometry.ik_fail_sec", ik_fail_sec_},
    };
    for (const auto & [name, value] : positives) {
        if (!std::isfinite(value) || value <= 0.0) {
            RCLCPP_ERROR(logger, "%s must be positive (got %f)", name, value);
            return false;
        }
    }
    // The tolerance is how far off the path the arm may be and still count as
    // on it. The lip path keeps `gap` from the receiver, so a tolerance that is
    // not well inside it spends the gap on tracking error.
    if (ik_tolerance_ >= lip_config_.gap) {
        RCLCPP_ERROR(logger,
            "geometry.ik_tolerance (%f m) must be under geometry.lip_gap (%f m): the arm "
            "would be allowed off the path by more than the clearance the path keeps",
            ik_tolerance_, lip_config_.gap);
        return false;
    }
    std::string why;
    if (!lip_config_.validate(why)) {
        RCLCPP_ERROR(logger, "geometry rejected: %s", why.c_str());
        return false;
    }
    if (lip_config_.hand_boxes.empty()) {
        RCLCPP_WARN(logger,
            "geometry.hand_boxes is empty, so the lip path keeps only the vessel itself out of "
            "the receiver. Under the side grasp the pour tips the vessel toward a jaw, which "
            "hangs just below the lip and past it: declare the jaws");
    }
    return true;
}

CallbackReturn PouringController::on_configure(const rclcpp_lifecycle::State & previous_state)
{
    if (!assign_parameters()) {
        return CallbackReturn::FAILURE;
    }
    if (FR5BaseController::on_configure(previous_state) != CallbackReturn::SUCCESS) {
        return CallbackReturn::FAILURE;
    }

    // The pour joint defaults to the last one: under the AG-95's side grasp its
    // axis runs through the grasp, so the vessel's tilt tracks it 1:1.
    pour_index_ = num_dof_ - 1;
    if (!pour_joint_.empty()) {
        const auto it = std::find(joint_names_.begin(), joint_names_.end(), pour_joint_);
        if (it == joint_names_.end()) {
            RCLCPP_ERROR(get_node()->get_logger(),
                "pour_joint '%s' is not one of this arm's joints", pour_joint_.c_str());
            return CallbackReturn::FAILURE;
        }
        pour_index_ = static_cast<int>(std::distance(joint_names_.begin(), it));
    }

    // An outlier bound in grams, derived from the fastest flow any configured
    // profile allows. Expressing it in grams rather than as a rate is what makes
    // it survive a dropped sample: a 3x margin covers two samples' worth.
    const double fastest = std::max({liquid_.free.max_flow_rate, liquid_.resistant.max_flow_rate,
                                     granular_.free.max_flow_rate, granular_.resistant.max_flow_rate});
    pour::ScaleFilter::Config filter_config;
    filter_config.max_step_grams = outlier_factor_ * fastest * scale_sample_period_;
    filter_config.rate_window_sec = 3.0 * scale_sample_period_;
    filter_config.history = 8;
    filter_.configure(filter_config);

    scale_sub_ = get_node()->create_subscription<sdl_cell_interfaces::msg::ScaleReading>(
        scale_topic_, rclcpp::SensorDataQoS(),
        [this](const sdl_cell_interfaces::msg::ScaleReading::SharedPtr msg) {
            pour::ScaleFilter::Sample sample;
            sample.grams = msg->grams;
            // Arrival time on THIS node's clock, not header.stamp. Every age and
            // settle window is measured against the controller_manager's update
            // time, and header.stamp is on whatever clock the driver ran --
            // wall time under a sim-time bringup, another PC's clock on a
            // multi-PC one -- so using it put two clocks in one subtraction.
            // It bought nothing, either: the HS-AA driver stamps when it has
            // PARSED a frame, after the 75 ms serial shift, and delivery is
            // 0.6 ms (measured), so the two differ by less than a control cycle.
            // The delay that matters is calibrated end to end as the profile's
            // transport_delay, on this clock.
            sample.stamp = get_node()->now().seconds();
            sample.stable = msg->stable;
            scale_buffer_.writeFromNonRT(sample);
        });

    vessel_sub_.reset();
    if (measured_geometry_) {
        vessel_buffer_.writeFromNonRT(VesselSample{});
        vessel_sub_ = get_node()->create_subscription<geometry_msgs::msg::PoseStamped>(
            vessel_pose_topic_, rclcpp::SensorDataQoS(),
            [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
                VesselSample sample;
                // Arrival, on this node's clock, for the same reason as the
                // scale -- and because what invalidates a pose is the arm
                // moving, which is timed on this clock too.
                sample.stamp = get_node()->now().seconds();
                sample.position = Eigen::Vector3d(msg->pose.position.x, msg->pose.position.y,
                                                  msg->pose.position.z);
                sample.frame_ok = msg->header.frame_id == vessel_pose_frame_;
                sample.finite = sample.position.allFinite();
                vessel_buffer_.writeFromNonRT(sample);
            });
    }
    attitude_sub_.reset();
    if (measured_geometry_ && level_enabled_ && !attitude_topic_.empty()) {
        attitude_buffer_.writeFromNonRT(AttitudeSample{});
        attitude_sub_ = get_node()->create_subscription<geometry_msgs::msg::PoseStamped>(
            attitude_topic_, rclcpp::SensorDataQoS(),
            [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
                AttitudeSample sample;
                sample.stamp = get_node()->now().seconds();
                const auto & q = msg->pose.orientation;
                const Eigen::Quaterniond rotation(q.w, q.x, q.y, q.z);
                sample.finite = std::isfinite(q.w) && std::isfinite(q.x) &&
                                std::isfinite(q.y) && std::isfinite(q.z) &&
                                rotation.norm() > 1e-6;
                // The tag's normal: its z, which is the vessel's axis.
                sample.axis = sample.finite
                    ? Eigen::Vector3d(rotation.normalized() * Eigen::Vector3d::UnitZ())
                    : Eigen::Vector3d::UnitZ();
                sample.frame_ok = msg->header.frame_id == vessel_pose_frame_;
                attitude_buffer_.writeFromNonRT(sample);
            });
    }

    action_server_ = std::make_shared<FR5PourActionServer>(
        get_node(), "/controller_action_server/pouring_controller", num_dof_);
    action_server_->init();
    PourBounds defaults;
    defaults.max_tilt_rate = 0.0;  // 0 means "whatever the material profile says"
    defaults.max_tilt = max_tilt_;
    defaults.tolerance = weight_tolerance_;
    defaults.timeout = 120.0;
    action_server_->set_defaults(defaults);
    action_server_->set_granular_max_tilt(granular_max_tilt_);
    action_server_->set_granular_timeout(granular_timeout_);
    // The lip path is laid out around the receiver, so a measured pour cannot
    // start without being told where it is.
    action_server_->set_requires_receiver(measured_geometry_);

    RCLCPP_INFO(get_node()->get_logger(),
        "PouringController configured: law '%s', pour joint '%s' (direction %+.0f), weight "
        "from %s, outlier bound %.1f g/sample, geometry '%s'%s%s",
        law_->name(), joint_names_[pour_index_].c_str(), pour_direction_, scale_topic_.c_str(),
        filter_config.max_step_grams, measured_geometry_ ? "measured" : "joint",
        measured_geometry_ ? ", vessel from " : "",
        measured_geometry_ ? vessel_pose_topic_.c_str() : "");
    if (measured_geometry_ && level_enabled_) {
        RCLCPP_INFO(get_node()->get_logger(),
            "The vessel is stood upright before each pour, by %s%s (up to %.3f rad, at %.3f "
            "rad/s, tag believed within %.3f rad of the kinematics)",
            attitude_topic_.empty() ? "the axis the grasp gave it" : "its tag on ",
            attitude_topic_.c_str(), level_config_.max_correction, level_rate_,
            level_config_.max_tag_disagreement);
    }
    if (!tilt_debug_pub_) {
        tilt_debug_pub_ = get_node()->create_publisher<std_msgs::msg::Float64MultiArray>(
            "~/tilt_debug", rclcpp::SystemDefaultsQoS());
        tilt_debug_rt_ = std::make_unique<
            realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>>(tilt_debug_pub_);
    }
    const std::string jerk_note =
        tilt_jerk_ > 0.0 ? std::to_string(tilt_jerk_) + " rad/s^3" : std::string("unbounded");
    RCLCPP_INFO(get_node()->get_logger(),
        "Tilt: accel %.3f rad/s^2, jerk %s; capped at what the IK (%.0f%% of max_delta_q) and "
        "the lip (%.0f%% of max_lip_speed) can follow; braked from 1x the IK tolerance behind, "
        "hard (%.0fx the accel) from %.2fx, stopped dead at %.1fx", tilt_accel_,
        jerk_note.c_str(), 100.0 * kIkRateShare, 100.0 * kLipRateShare, kEmergencyBrake,
        ik_hold_ratio_, kDeadHoldRatio);
    return CallbackReturn::SUCCESS;
}

CallbackReturn PouringController::on_activate(const rclcpp_lifecycle::State & previous_state)
{
    if (FR5BaseController::on_activate(previous_state) != CallbackReturn::SUCCESS) {
        return CallbackReturn::FAILURE;
    }
    // From the held command, not the measurement: the controller this one takes
    // over from left a position on the interface, and starting from the measured
    // (drooped) one would step the command by the droop on the first cycle --
    // with a full vessel in the gripper.
    q_ref_ = FR5BaseController::held_command_position();
    phase_ = Phase::Idle;
    tilt_ = 0.0;
    shaker_.reset();
    shake_amplitude_ = 0.0;
    sent_pour_ = std::numeric_limits<double>::quiet_NaN();
    sent_tap_ = 0.0;
    filter_.reset();
    last_pushed_stamp_ = 0.0;
    scale_buffer_.writeFromNonRT(pour::ScaleFilter::Sample{});
    // Whatever arrived before now was measured under another controller, which
    // may have been moving the arm.
    vessel_buffer_.writeFromNonRT(VesselSample{});
    attitude_buffer_.writeFromNonRT(AttitudeSample{});
    quiet_since_ = get_node()->now().seconds();
    law_started_ = false;
    return CallbackReturn::SUCCESS;
}

CallbackReturn PouringController::on_deactivate(const rclcpp_lifecycle::State & previous_state)
{
    // A goal that outlives its controller can never finish: nothing is left to
    // run the law, and the caller would wait for a result that cannot come.
    if (action_server_) {
        action_server_->abort_active_goal(
            "the pouring controller was deactivated mid-pour", last_grams_, peak_tilt_, elapsed_);
    }
    phase_ = Phase::Idle;
    return FR5BaseController::on_deactivate(previous_state);
}

void PouringController::write_command(const Eigen::VectorXd & q_cmd)
{
    if (q_ref_.size() == q_cmd.size() && (q_cmd - q_ref_).cwiseAbs().maxCoeff() > 1e-9) {
        quiet_since_ = now_;
    }
    q_ref_ = q_cmd;
    send(q_cmd);
}

void PouringController::hold_reference()
{
    send(q_ref_);
}

void PouringController::send(const Eigen::VectorXd & q)
{
    Eigen::VectorXd out = q;
    if (pour_index_ >= 0 && pour_index_ < out.size()) {
        const double untapped = out(pour_index_);
        const double tap = goal_direction_ * shaker_.offset();
        double sent = untapped + tap;
        // Bounded only while a tap is on or still being unwound, so a command
        // with no tap in it is sent exactly as given. The bound is the tilt's
        // and the tap's together: one step of the joint, whatever it is made of.
        if ((tap != 0.0 || sent_tap_ != 0.0) && std::isfinite(sent_pour_)) {
            const double step = pour_speed_limit(nominal_period_) * nominal_period_;
            sent = std::clamp(sent, sent_pour_ - step, sent_pour_ + step);
        }
        out(pour_index_) = sent;
        clamp_to_joint_limits(out);
        sent_pour_ = out(pour_index_);
        sent_tap_ = sent_pour_ - untapped;
        if (std::abs(sent_tap_) < 1e-12) {
            sent_tap_ = 0.0;
        }
    }
    for (int i = 0; i < num_dof_; ++i) {
        command_interfaces_[i].set_value(out(i));
    }
    state_.q_des = out;
}

void PouringController::begin_untilt(bool succeeded, const std::string & reason)
{
    if (phase_ == Phase::Untilt) {
        return;
    }
    phase_ = Phase::Untilt;
    pending_success_ = succeeded;
    pending_reason_ = reason;
    if (!succeeded) {
        RCLCPP_WARN(get_node()->get_logger(),
            "Pour ending without reaching the target (%s); returning the vessel to the "
            "attitude it was carried in first", reason.c_str());
    }
}

void PouringController::finish()
{
    const pour::PourReport none;
    const auto & report = law_started_ ? law_->report() : none;
    if (pending_success_) {
        action_server_->succeed(report.poured_grams, peak_tilt_, elapsed_, report.trim_pulses,
                                report.measured_afterflow);
    } else {
        action_server_->abort_active_goal(pending_reason_, report.poured_grams, peak_tilt_,
                                          elapsed_, report.trim_pulses,
                                          report.measured_afterflow);
    }
    phase_ = Phase::Idle;
}

pour::PourRequest PouringController::make_request() const
{
    const auto & bounds = action_server_->bounds();
    pour::PourRequest request;
    request.target_grams = bounds.target_grams;
    request.container_grams = bounds.container_grams;
    request.tolerance = bounds.tolerance;
    request.timeout = bounds.timeout;
    request.max_tilt = bounds.max_tilt;
    request.max_tilt_rate = bounds.max_tilt_rate;
    request.material = bounds.material == 1 ? MaterialClass::Granular : MaterialClass::Liquid;
    request.flow_index = bounds.flow_index;
    request.max_back_tilt = max_back_tilt_;
    request.tilt_accel = tilt_accel_;
    // The filter fits its flow over three sample periods; the flow it reports
    // belongs to the middle of that window.
    request.flow_fit_lag = 1.5 * scale_sample_period_;
    return request;
}

bool PouringController::start_goal(double now)
{
    // A granular bed and a liquid are not tipped the same way. Tipped toward
    // the fingertips -- the reference recordings' way, and water's -- the tilt
    // is the wrist pitching, which on this arm is j2 + j3 + j4, and sugar
    // (2026-09-28) needed more of it than j4's range has left: the pour
    // stopped at 77.7 deg with j4 at its limit and a third of the goal still
    // in the beaker. Rolled toward a jaw instead, the tilt is the pour joint's
    // own, and j6 has +-175 deg.
    const bool granular =
        action_server_->bounds().material == PourAction::Goal::MATERIAL_GRANULAR;
    const bool roll = granular && granular_tilt_about_ == "pour_joint";
    // Water toward the fingertips, whatever the goal's reference shows: a
    // planner's recorded pour may roll the wrist (the two-beaker cell's turned
    // j6 by -85 deg, 2026-09-29), and a rolled liquid goes over the jaw's side.
    const bool fingertips = !granular && liquid_tilt_about_ == "fingertips";
    const double configured =
        (granular && granular_pour_direction_ != 0.0) ? granular_pour_direction_ : pour_direction_;
    const int asked = action_server_->bounds().pour_direction;
    goal_direction_ = asked != 0 ? static_cast<double>(asked) : configured;
    if (asked != 0 && goal_direction_ != configured) {
        RCLCPP_INFO(get_node()->get_logger(),
            "This goal pours with direction %+.0f, not the configured %+.0f",
            goal_direction_, configured);
    }

    // How the goal says to tip: the axis the EE turns about from here to the
    // reference configuration, in this controller's own kinematics.
    reference_axis_valid_ = false;
    const auto & reference = action_server_->bounds().pour_reference_joints;
    if (!reference.empty() && roll) {
        RCLCPP_INFO(get_node()->get_logger(),
            "Granular: tipping about the pour joint (direction %+.0f), not the goal's reference "
            "configuration (granular.tilt_about: pour_joint)", goal_direction_);
    }
    if (fingertips) {
        if (!measured_geometry_) {
            abort_before_motion("the pour was not started: liquid.tilt_about: fingertips tips "
                                "the vessel about the jaws, which needs pour_geometry: measured");
            return false;
        }
        Eigen::VectorXd q_now = state_.q;
        q_now.head(num_dof_) = q_ref_;
        pinocchio::SE3 H_now;
        Eigen::MatrixXd J_unused;
        compute_arm_kinematics(q_now, H_now, J_unused);
        double heading = 0.0;
        std::string why;
        if (!fingertips_axis(H_now, reference_axis_ee_, heading, why)) {
            abort_before_motion("the pour was not started: " + why);
            return false;
        }
        reference_axis_valid_ = true;
        RCLCPP_INFO(get_node()->get_logger(),
            "Liquid: tipping toward the fingertips (heading %.0f deg), not the goal's reference "
            "configuration (liquid.tilt_about: fingertips)", heading * 180.0 / M_PI);
    }
    fingertips_ = fingertips;
    if (!reference.empty() && !roll && !fingertips) {
        Eigen::VectorXd q_now = state_.q;
        q_now.head(num_dof_) = q_ref_;
        Eigen::VectorXd q_deep = state_.q;
        for (int i = 0; i < num_dof_; ++i) {
            q_deep(i) = reference[i];
        }
        pinocchio::SE3 H_now, H_deep;
        Eigen::MatrixXd J_now, J_unused;
        compute_arm_kinematics(q_now, H_now, J_now);
        compute_arm_kinematics(q_deep, H_deep, J_unused);
        const Eigen::Vector3d turn =
            pinocchio::log3(H_deep.rotation() * H_now.rotation().transpose());
        const double angle = turn.norm();
        // Less than this is not a pour; its axis would be noise.
        constexpr double kMinReferenceTilt = 0.2;
        if (!std::isfinite(angle) || angle < kMinReferenceTilt) {
            std::ostringstream os;
            os << "the pour was not started: its reference configuration turns the EE only "
               << std::fixed << std::setprecision(3) << angle << " rad from where the goal "
                  "starts, which does not show a pour";
            abort_before_motion(os.str());
            return false;
        }
        const Eigen::Vector3d axis_world = turn / angle;
        if (measured_geometry_) {
            reference_axis_ee_ = H_now.rotation().transpose() * axis_world;
            reference_axis_valid_ = true;
        } else {
            // The pour joint alone can only tip about its own axis.
            const Eigen::Vector3d joint_axis =
                (H_now.rotation() * J_now.col(pour_index_).tail<3>()).normalized();
            const double c = joint_axis.dot(axis_world);
            if (std::abs(c) < 0.8) {
                std::ostringstream os;
                os << "the pour was not started: it tips the vessel about an axis "
                   << std::fixed << std::setprecision(2)
                   << std::acos(std::min(1.0, std::abs(c))) << " rad from the pour joint's, "
                      "which turning that joint alone cannot do. pour_geometry: measured can";
                abort_before_motion(os.str());
                return false;
            }
            goal_direction_ = c > 0.0 ? 1.0 : -1.0;
        }
    }
    theta_start_ = q_ref_(pour_index_);
    tilt_ = 0.0;
    tilt_ramp_.reset();
    logged_stops_ = 0;
    stop_graded_ = true;
    law_target_tilt_ = std::numeric_limits<double>::quiet_NaN();
    reach_ = std::numeric_limits<double>::infinity();
    last_law_rate_ = 0.0;
    shake_amplitude_ = 0.0;
    ik_blocked_since_ = -1.0;
    peak_tilt_ = 0.0;
    elapsed_ = 0.0;
    feedback_accumulator_ = 0.0;
    last_phase_ = 0;
    filter_.reset();
    last_pushed_stamp_ = 0.0;
    law_started_ = false;
    pending_success_ = false;
    pending_reason_.clear();

    if (measured_geometry_) {
        phase_ = Phase::Measure;
        measure_started_ = now;
        q_start_ = q_ref_;
        held_measured_ = false;
        level_s_ = 0.0;
        level_ramp_.reset();
        level_note_.clear();
        align_s_ = 0.0;
        align_ramp_.reset();
        inset_cmd_ = 0.0;
        ik_lagging_ = false;
        ik_lag_ratio_ = 0.0;
        ik_bad_since_ = -1.0;
        return true;
    }
    start_pour(now);
    return true;
}

void PouringController::start_pour(double now)
{
    pour::PourRequest request = make_request();
    if (measured_geometry_) {
        // Past this the lip would have to leave the mouth to keep the jaws off
        // the rim, and the stream would leave with it.
        request.max_tilt = std::min(request.max_tilt, lip_path_.last_landing_tilt());
    }
    tilt_bound_ = request.max_tilt;
    tap_bound_ = request.max_tilt;
    law_->begin(request, now);
    law_started_ = true;
    phase_ = Phase::Pouring;
}

double PouringController::step_law(double now)
{
    if (action_server_->is_canceling()) {
        law_->cancel();
    }
    pour::PourObservation obs;
    obs.now = now;
    obs.has_reading = filter_.has_sample();
    obs.scale_fresh = obs.has_reading && filter_.age(now) <= scale_timeout_;
    obs.grams = filter_.grams();
    obs.flow_rate = filter_.flow_rate();
    obs.sample_stamp = filter_.stamp();
    obs.settled = filter_.settled(now, law_->settle_hold());
    obs.tilt = tilt_;
    obs.consecutive_rejects = filter_.consecutive_rejects();
    obs.last_rejected_step = filter_.last_rejected_step();
    obs.reach = reach_;

    const pour::PourCommand cmd = law_->update(obs);
    last_law_rate_ = cmd.tilt_rate;
    shake_amplitude_ = cmd.finished ? 0.0 : cmd.shake_amplitude;
    shake_period_ = cmd.shake_period;
    last_phase_ = static_cast<std::uint8_t>(cmd.phase);
    // One line per stop, and one per settle after it: what the forecast said
    // was still coming, then what actually came. That pair is the only grade
    // the forecast gets on the rig.
    const auto & report = law_->report();
    if (report.stops > logged_stops_) {
        logged_stops_ = report.stops;
        stop_graded_ = false;
        RCLCPP_INFO(get_node()->get_logger(),
                    "Stop %d at %.2f g landed: forecast %.2f g in the air + %.2f g on the way "
                    "down + %.2f g tail = %.2f g",
                    report.stops, report.stop_landed, report.stop_in_flight,
                    report.stop_during_retract, report.stop_afterflow,
                    report.stop_landed + report.stop_in_flight + report.stop_during_retract +
                        report.stop_afterflow);
    }
    if (report.stops > 0 && !stop_graded_ && cmd.phase != pour::PourPhase::Retract &&
        cmd.phase != pour::PourPhase::Settle) {
        stop_graded_ = true;
        RCLCPP_INFO(get_node()->get_logger(),
                    "Stop %d settled at %.2f g: %.2f g came after it, forecast %.2f g",
                    report.stops, report.poured_grams, report.poured_grams - report.stop_landed,
                    report.stop_in_flight + report.stop_during_retract + report.stop_afterflow);
    }
    law_target_tilt_ = cmd.target_tilt;
    if (cmd.finished) {
        begin_untilt(cmd.success, cmd.message);
        return 0.0;
    }
    return cmd.tilt_rate;
}

double PouringController::shaped_law_step(double rate, double dt)
{
    if (std::isfinite(law_target_tilt_)) {
        return tilt_ramp_.reach(tilt_, law_target_tilt_, std::abs(rate), tilt_accel_, dt);
    }
    return tilt_ramp_.follow(rate, tilt_accel_, dt);
}

void PouringController::publish_feedback_if_due(double dt)
{
    feedback_accumulator_ += dt;
    if (feedback_accumulator_ >= feedback_period_) {
        feedback_accumulator_ -= feedback_period_;
        const double poured = law_started_ ? last_grams_ - law_->baseline_grams() : 0.0;
        action_server_->publish_feedback(poured, tilt_, elapsed_, last_flow_, last_phase_);
    }
}

controller_interface::return_type PouringController::update(
    const rclcpp::Time & time, const rclcpp::Duration & period)
{
    if (FR5BaseController::update(time, period) != controller_interface::return_type::OK) {
        return controller_interface::return_type::ERROR;
    }

    if (q_ref_.size() != num_dof_) {
        q_ref_ = state_.q.head(num_dof_);
    }
    now_ = time.seconds();

    // Before anything is sent this cycle, idle included: a tap that is under
    // way when the pour stops still has to come back down.
    nominal_period_ = nominal_period(period);
    // A tap tips the vessel past the tilt it is at, so it may only go as far
    // as the tilt itself could: at the lip path's bound -- where a jaw comes
    // onto the rim -- the taps fall silent. On the rig (2026-09-28) taps held
    // at that bound put a jaw on the receiver. The arm's reach is not such a
    // bound: a joint at its limit is not the vessel at the rim, and the taps
    // are j6's alone.
    const double tap_room = std::max(0.0, tap_bound_ - kTapMargin - tilt_);
    shaker_.step(phase_ == Phase::Pouring ? std::min(shake_amplitude_, tap_room) : 0.0,
                 shake_period_, kTapRateShare * pour_speed_limit(nominal_period_),
                 nominal_period_);

    const bool running = action_server_ && action_server_->is_running();
    const bool returning = phase_ == Phase::Untilt || phase_ == Phase::Unalign ||
                           phase_ == Phase::Unlevel || phase_ == Phase::Return;
    if (!running && !returning) {
        // Idle: freeze the reference. Re-deriving a hold from the measured
        // position would feed servo droop back into the command.
        phase_ = Phase::Idle;
        hold_reference();
        return controller_interface::return_type::OK;
    }

    const double dt = nominal_period(period);
    const double now = now_;

    // Take whatever the subscription left, but only once. Re-pushing the same
    // sample every cycle would make the least-squares fit see 125 identical
    // points per reading and report a flow rate of zero throughout a pour.
    const pour::ScaleFilter::Sample sample = *scale_buffer_.readFromRT();
    if (sample.stamp > last_pushed_stamp_) {
        last_pushed_stamp_ = sample.stamp;
        filter_.push(sample);
    }

    if (phase_ == Phase::Idle && !start_goal(now)) {
        return controller_interface::return_type::OK;
    }

    elapsed_ += dt;
    last_grams_ = filter_.grams();
    last_flow_ = filter_.flow_rate();

    if (measured_geometry_) {
        return update_measured(now, dt);
    }

    const double tilt_before = tilt_;
    double step = 0.0;

    if (phase_ == Phase::Pouring) {
        const double rate = step_law(now);
        if (phase_ == Phase::Pouring) {
            step = shaped_law_step(rate, dt);
        }
    }

    if (phase_ == Phase::Untilt) {
        last_phase_ = static_cast<std::uint8_t>(PourPhase::Done);
        const double bound = std::abs(goal_direction_) * max_delta_q_ / std::max(dt, 1e-9);
        const double rate = std::min(law_->return_tilt_rate(), bound);
        step = tilt_ramp_.reach(tilt_, 0.0, rate, tilt_accel_, dt);
        if (tilt_ + step == 0.0) {
            tilt_ = 0.0;
            Eigen::VectorXd q_cmd = q_ref_;
            q_cmd(pour_index_) = theta_start_;
            clamp_to_joint_limits(q_cmd);
            write_command(q_cmd);
            finish();
            return controller_interface::return_type::OK;
        }
    }

    tilt_ += step;

    // Tilt range, measured from the attitude the pour started in, and BOTH ends
    // of it. The laws have their own bounds and normally stay inside; this is
    // the one that holds if a law is ever wrong -- a park lowered too far, or a
    // law driving a negative error backwards through upright. Bounding only the
    // pour side (what this did after the rewrite) left the other direction
    // limited by nothing short of the joint limit.
    if (phase_ == Phase::Pouring) {
        tilt_ = std::clamp(tilt_, -max_back_tilt_, tilt_bound_);
    }
    peak_tilt_ = std::max(peak_tilt_, std::abs(tilt_));

    Eigen::VectorXd q_cmd = q_ref_;
    q_cmd(pour_index_) = theta_start_ + goal_direction_ * tilt_;
    clamp_to_joint_limits(q_cmd);

    // Per-cycle command bound. The trajectory the law produces is already rate
    // limited, but this also covers the joint-limit clamp above and anything a
    // future edit puts between the two.
    Eigen::VectorXd delta = q_cmd - q_ref_;
    delta = delta.array().max(-max_delta_q_).min(max_delta_q_);
    q_cmd = q_ref_ + delta;

    if (!q_cmd.allFinite()) {
        action_server_->abort_active_goal(
            "the pour command went non-finite; holding the last finite command",
            last_grams_, peak_tilt_, elapsed_, law_->report().trim_pulses,
            law_->report().measured_afterflow);
        phase_ = Phase::Idle;
        hold_reference();
        return controller_interface::return_type::OK;
    }

    write_command(q_cmd);
    // The tilt follows the CLAMPED command, so a joint limit or the per-cycle
    // bound cannot leave the planner reasoning about an angle the arm was never
    // asked to reach -- nor the ramp accelerating from a speed it never had.
    tilt_ = goal_direction_ * (q_cmd(pour_index_) - theta_start_);
    tilt_ramp_.observed(tilt_ - tilt_before, dt);

    publish_feedback_if_due(dt);
    return controller_interface::return_type::OK;
}

// ---------------------------------------------------------------- measured

bool PouringController::vessel_sample_usable(const VesselSample & sample, double now) const
{
    return sample.stamp > 0.0 && sample.frame_ok && sample.finite &&
           now - sample.stamp <= vessel_pose_max_age_ &&
           sample.stamp >= quiet_since_ + vessel_settle_sec_;
}

std::string PouringController::vessel_sample_problem(const VesselSample & sample, double now) const
{
    std::ostringstream os;
    if (sample.stamp <= 0.0) {
        os << "nothing has arrived on " << vessel_pose_topic_ << " since the controller was "
           << "activated. Is cho_object_pose running with the held vessel's marker in its "
              "table, and can a side camera see the marker from where the vessel is held?";
    } else if (!sample.frame_ok) {
        os << "the poses on " << vessel_pose_topic_ << " are not in '" << vessel_pose_frame_
           << "', and nothing here transforms them";
    } else if (!sample.finite) {
        os << "the latest pose on " << vessel_pose_topic_ << " is not finite";
    } else if (now - sample.stamp > vessel_pose_max_age_) {
        os << "the latest pose on " << vessel_pose_topic_ << " is " << std::fixed
           << std::setprecision(1) << now - sample.stamp << " s old: the marker has gone out "
              "of view, or the pose node has stopped trusting it";
    } else if (sample.stamp < quiet_since_ + vessel_settle_sec_) {
        os << "every pose so far may have been averaged over the arm's last move; none has "
              "arrived vessel_settle_sec after it stopped";
    }
    return os.str();
}

bool PouringController::attitude_usable(const AttitudeSample & sample, double now) const
{
    return sample.stamp > 0.0 && sample.frame_ok && sample.finite &&
           now - sample.stamp <= vessel_pose_max_age_ &&
           sample.stamp >= quiet_since_ + vessel_settle_sec_;
}

bool PouringController::fingertips_axis(const pinocchio::SE3 & H, Eigen::Vector3d & axis_ee,
                                        double & heading, std::string & why) const
{
    // The jaws' centre line on the horizontal, toward the fingertips, as the
    // lip path finds it.
    const Eigen::Vector3d up = Eigen::Vector3d::UnitZ();
    const Eigen::Vector3d closing = (H.rotation() * lip_config_.closing_axis_in_ee).normalized();
    const Eigen::Vector3d approach = (H.rotation() * lip_config_.approach_axis_in_ee).normalized();
    Eigen::Vector3d along = up.cross(closing);
    along.z() = 0.0;
    if (along.norm() < 1e-6) {
        why = "the jaws close vertically, so there is no fingertip side to tip toward";
        return false;
    }
    along.normalize();
    if (along.dot(approach) < 0.0) {
        along = -along;
    }
    // The horizontal across the jaws, signed so a positive tilt brings the
    // fingertip side of the vessel down.
    Eigen::Vector3d axis_world = up.cross(along);
    if (axis_world.cross(along).dot(up) > 0.0) {
        axis_world = -axis_world;
    }
    axis_ee = H.rotation().transpose() * axis_world;
    heading = std::atan2(along.y(), along.x());
    return true;
}

bool PouringController::plan_pour_geometry(std::string & why)
{
    // The grasp, from the command the arm is holding and the vessel held_ has.
    Eigen::VectorXd q_full = state_.q;
    q_full.head(num_dof_) = q_ref_;
    pinocchio::SE3 H;
    Eigen::MatrixXd J;
    compute_arm_kinematics(q_full, H, J);
    // Toward the fingertips is a direction of the jaws on the horizontal, and
    // levelling may have turned the EE: found again from where it is now.
    if (fingertips_) {
        double heading = 0.0;
        if (!fingertips_axis(H, reference_axis_ee_, heading, why)) {
            return false;
        }
        reference_axis_valid_ = true;
    }
    // The local Jacobian's angular column for a revolute joint is its axis
    // in the EE frame.
    const Eigen::Vector3d axis = reference_axis_valid_
        ? Eigen::Vector3d(H.rotation() * reference_axis_ee_)
        : Eigen::Vector3d(goal_direction_ * (H.rotation() * J.col(pour_index_).tail<3>()));
    // Rolled toward a jaw, the jaw hangs under the lip: on this gripper its
    // lower edge is 48 mm below it and 20 mm out, and held at 45-58 mm it
    // came onto the receiver's rim by 0.26 rad (2026-09-28) -- far short of
    // sugar's ~1.1 rad onset. A granular pour can hold its lip higher; what
    // it pours does not splash.
    pour::LipPathConfig lip_config = lip_config_;
    const bool granular =
        action_server_->bounds().material == PourAction::Goal::MATERIAL_GRANULAR;
    if (granular && granular_lip_inset_ > 0.0) {
        lip_config.inset = granular_lip_inset_;
    }
    if (granular && granular_lip_height_ > 0.0) {
        lip_config.lip_height = granular_lip_height_;
        lip_config.max_height = std::max(lip_config.max_height, granular_lip_height_);
    }
    // The stream has to land in the receiver from the first tilt anything
    // can come out at -- and not before it: nothing pours below the seek's
    // fast range, which for a granular bed runs its angle of repose past
    // the liquid's. Asked to land from 0.5 rad, the lip path lifted a
    // rolled beaker's lip from 45 to 58 mm to catch a stream sugar never
    // makes there, and at 58 the jaw under the lip met the rim by 0.26
    // rad. Asked from 0.85 it keeps 45 mm and pours 0.72-1.75 rad.
    const pour::PourLimits & limits =
        (granular ? granular_ : liquid_).at(action_server_->bounds().flow_index);
    lip_config.landing_by_tilt = std::max(
        lip_config.landing_by_tilt, planner_config_.seek_fast_until + limits.repose_angle);
    // The receiver's radius is the cell's; where it stands is the goal's.
    pour::Receiver receiver = receiver_;
    const auto & rim = action_server_->bounds().receiver_rim_center;
    if (rim.size() != 3) {
        why = "the goal did not say where the receiver's rim is";
        return false;
    }
    receiver.rim_center = Eigen::Vector3d(rim[0], rim[1], rim[2]);
    // Where the receiver stands is perception's; how high its rim is, is
    // not yet. beaker2's tag put its bottom at z 0.080 (2026-09-29), 20 mm
    // under where it stands, and every height the pour holds is measured
    // from the rim: the whole pour came down with it.
    if (receiver_rim_z_ > 0.0) {
        if (std::abs(rim[2] - receiver_rim_z_) > 1e-4) {
            RCLCPP_INFO(get_node()->get_logger(),
                "Receiver rim height fixed at %.4f m (geometry.receiver_rim_z); the goal "
                "said %.4f", receiver_rim_z_, rim[2]);
        }
        receiver.rim_center.z() = receiver_rim_z_;
    }
    const Eigen::Isometry3d ee = to_isometry(H);
    if (!lip_path_.plan(ee, axis, held_.bottom(ee), action_server_->bounds().max_tilt, vessel_,
                        receiver, lip_config, why)) {
        return false;
    }
    // A new path: the look-ahead's cells are filled again as they are asked for.
    lip_table_lo_ = -max_back_tilt_ - kLipLookahead;
    const double table_hi = action_server_->bounds().max_tilt + kLipLookahead;
    lip_table_.assign(
        static_cast<std::size_t>(std::ceil((table_hi - lip_table_lo_) / kLipTableStep)) + 1,
        std::numeric_limits<double>::quiet_NaN());
    RCLCPP_INFO(get_node()->get_logger(),
                "Pour geometry (%s%s%s), receiver rim at (%.4f, %.4f, %.4f): %s",
                action_server_->bounds().grasp_joints.empty() ? "marker measured now"
                                                              : "grasp measured before the goal",
                level_note_.empty() ? "" : "; vessel ", level_note_.c_str(),
                receiver.rim_center.x(), receiver.rim_center.y(), receiver.rim_center.z(),
                lip_path_.summary().c_str());
    return true;
}

void PouringController::abort_before_motion(const std::string & reason)
{
    RCLCPP_WARN(get_node()->get_logger(), "Pour refused before anything moved: %s",
                reason.c_str());
    action_server_->abort_active_goal(reason, 0.0, 0.0, elapsed_);
    phase_ = Phase::Idle;
    hold_reference();
}

void PouringController::fail_geometric(const std::string & reason, Phase via)
{
    if (phase_ == Phase::Unalign || phase_ == Phase::Unlevel || phase_ == Phase::Return) {
        // Already on the way back with a reason of its own. Only a failure to
        // follow the way back changes anything: it goes to joint space.
        if (via == Phase::Return) {
            phase_ = Phase::Return;
        }
        return;
    }
    const bool poured = phase_ == Phase::Untilt && pending_success_;
    pending_reason_ = poured ? "the pour reached its target, but then " + reason : reason;
    pending_success_ = false;
    RCLCPP_WARN(get_node()->get_logger(),
        "Pour ending without reaching the target (%s); returning the arm to where the goal "
        "started first", pending_reason_.c_str());
    ik_lagging_ = false;
    ik_bad_since_ = -1.0;
    ik_blocked_since_ = -1.0;
    phase_ = via;
}

bool PouringController::track(const Eigen::Isometry3d & target, double now, std::string & why)
{
    Eigen::VectorXd q_full = state_.q;
    q_full.head(num_dof_) = q_ref_;
    pinocchio::SE3 H;
    Eigen::MatrixXd J;
    compute_arm_kinematics(q_full, H, J);

    // The task point is the LIP, not the EE origin. The origin sits ~0.3 m
    // from it, so a rotation lag the origin barely shows moves the lip by
    // millimetres -- 6 mm at the 0.02 rad the rotation tolerance allows,
    // measured against the riser recordings, which pour by turning j4 1.4x as
    // fast as the vessel tips and so run into max_delta_q.
    const Eigen::Vector3d r = lip_path_.lip_in_ee();
    const auto point_of = [&r](const Eigen::Matrix3d & R, const Eigen::Vector3d & p) {
        return Eigen::Vector3d(p + R * r);
    };
    // Local-frame task error against the COMMAND, like the task-space IK: the
    // measured joints never enter it, so servo noise is not fed back.
    Eigen::Matrix<double, 6, 1> error;
    error.head<3>() = H.rotation().transpose() *
        (point_of(target.linear(), target.translation()) - point_of(H.rotation(), H.translation()));
    error.tail<3>() = pinocchio::log3(H.rotation().transpose() * target.linear());
    // A point r off the frame moves at v + w x r = v - [r]x w.
    Eigen::Matrix3d skew_r;
    skew_r << 0.0, -r.z(), r.y(), r.z(), 0.0, -r.x(), -r.y(), r.x(), 0.0;
    Eigen::MatrixXd J_task = J;
    J_task.topRows<3>() -= skew_r * J.bottomRows<3>();
    Eigen::Matrix<double, 6, 6> JJt = J_task * J_task.transpose();
    JJt.diagonal().array() += ik_lambda_ * ik_lambda_;
    ik_JJt_ldlt_.compute(JJt);
    ik_J_task_ = J_task;
    ik_R_ = H.rotation();
    ik_solve_cached_ = true;
    Eigen::VectorXd dq = J_task.transpose() * ik_JJt_ldlt_.solve(error);
    if (!dq.allFinite()) {
        why = "the lip path's IK solve went non-finite";
        return false;
    }
    // Scaled as a whole, not clamped joint by joint: clamping one joint and
    // not the others bends the step off the path it was solved for.
    const double biggest = dq.cwiseAbs().maxCoeff();
    if (biggest > max_delta_q_) {
        dq *= max_delta_q_ / biggest;
    }
    Eigen::VectorXd q_cmd = q_ref_ + dq;
    clamp_to_joint_limits(q_cmd);

    q_full.head(num_dof_) = q_cmd;
    compute_arm_kinematics(q_full, H, J);
    const double pos_err = (point_of(target.linear(), target.translation()) -
                            point_of(H.rotation(), H.translation())).norm();
    const double rot_err =
        pinocchio::log3(H.rotation().transpose() * target.linear()).norm();
    if (!q_cmd.allFinite() || !std::isfinite(pos_err) || !std::isfinite(rot_err)) {
        why = "the lip path's IK produced a non-finite command";
        return false;
    }
    write_command(q_cmd);

    ik_lagging_ = pos_err > ik_tolerance_ || rot_err > ik_rot_tolerance_;
    ik_lag_ratio_ = std::max(pos_err / ik_tolerance_, rot_err / ik_rot_tolerance_);
    if (ik_lag_ratio_ >= kBlockedRatio) {
        if (ik_blocked_since_ < 0.0) {
            ik_blocked_since_ = now;
        }
    } else {
        ik_blocked_since_ = -1.0;
    }
    if (!ik_lagging_) {
        ik_bad_since_ = -1.0;
        return true;
    }
    if (ik_bad_since_ < 0.0) {
        ik_bad_since_ = now;
    }
    if (now - ik_bad_since_ > ik_fail_sec_) {
        std::ostringstream os;
        os << "the arm could not follow the lip path: " << std::fixed << std::setprecision(1)
           << pos_err * 1000.0 << " mm and " << std::setprecision(3) << rot_err
           << " rad off it for longer than ik_fail_sec. A joint limit or a singularity is in "
              "the way of tipping the vessel about its lip from this pose";
        why = os.str();
        return false;
    }
    return true;
}

void PouringController::publish_tilt_debug(double now, double proposed)
{
    if (!tilt_debug_rt_ || !tilt_debug_rt_->trylock()) {
        return;
    }
    auto & data = tilt_debug_rt_->msg_.data;
    data.resize(12);
    data[0] = now;
    data[1] = static_cast<double>(last_phase_);
    data[2] = tilt_;
    data[3] = proposed;
    data[4] = tilt_ramp_.velocity();
    data[5] = tilt_ramp_.acceleration();
    data[6] = ik_lag_ratio_;
    data[7] = static_cast<double>(hold_reason_);
    data[8] = inset_cmd_;
    data[9] = ik_speed_scale();
    data[10] = std::isfinite(tilt_cap_ik_) ? tilt_cap_ik_ : -1.0;
    data[11] = std::isfinite(tilt_cap_lip_) ? tilt_cap_lip_ : -1.0;
    tilt_debug_rt_->unlockAndPublish();
}

double PouringController::ik_speed_scale() const
{
    // The path used to run at full speed until the IK fell a tolerance behind
    // and then stop dead until it caught up. At the 0.15 rad/s the vessel is
    // tipped back upright at, that was a cycle every 0.5-0.9 s on the rig
    // (2026-09-28): accelerate, trip, chatter on and off for 60-120 ms at the
    // control rate, restart from rest -- a knock the operator could hear.
    // Slowing down as the residual grows lets the path settle at the speed
    // the IK can follow; the hard stop is left for a residual that grows
    // regardless.
    return std::clamp((1.0 - ik_lag_ratio_) / (1.0 - kSlowFrom), 0.0, 1.0);
}

double PouringController::joints_per_tilt(double tilt, double inset) const
{
    if (!ik_solve_cached_) {
        return 0.0;
    }
    // The task error a step of 2 kPathEps along the path would give the IK,
    // taken the way track() takes it -- the lip point and the rotation, in the
    // frame of the last solve -- and solved with that solve's own matrices.
    const Eigen::Isometry3d a = lip_path_.ee_pose(tilt - kPathEps, inset);
    const Eigen::Isometry3d b = lip_path_.ee_pose(tilt + kPathEps, inset);
    const Eigen::Vector3d r = lip_path_.lip_in_ee();
    Eigen::Matrix<double, 6, 1> step;
    step.head<3>() = ik_R_.transpose() *
        ((b.translation() + b.linear() * r) - (a.translation() + a.linear() * r));
    step.tail<3>() = pinocchio::log3(Eigen::Matrix3d(a.linear().transpose() * b.linear()));
    const Eigen::VectorXd dq = ik_J_task_.transpose() * ik_JJt_ldlt_.solve(step);
    if (!dq.allFinite()) {
        return 0.0;
    }
    return dq.cwiseAbs().maxCoeff() / (2.0 * kPathEps);
}

double PouringController::lip_window_end(double from, double direction) const
{
    // No further than the motion goes: upright on the way back, the law's
    // target when it has one, the tilt's range otherwise.
    double end = direction > 0.0 ? tilt_bound_ : -max_back_tilt_;
    if (phase_ == Phase::Untilt) {
        end = 0.0;
    } else if (std::isfinite(law_target_tilt_)) {
        end = law_target_tilt_;
    }
    if ((end - from) * direction <= 0.0) {
        return from;
    }
    return from + std::copysign(std::min(kLipLookahead, std::abs(end - from)), direction);
}

double PouringController::lip_limit_cached(double tilt) const
{
    const double x = (tilt - lip_table_lo_) / kLipTableStep;
    if (lip_table_.empty() || !(x >= 0.0) ||
        x > static_cast<double>(lip_table_.size() - 1)) {
        return lip_path_.inset_limit(tilt);
    }
    const auto i = static_cast<std::size_t>(std::lround(x));
    double & cell = lip_table_[i];
    if (std::isnan(cell)) {
        cell = lip_path_.inset_limit(lip_table_lo_ + static_cast<double>(i) * kLipTableStep);
    }
    return cell;
}

double PouringController::lip_ahead(double from, double direction) const
{
    double lowest = lip_path_.inset_limit(from);
    if (std::abs(direction) < 1e-12) {
        return lowest;
    }
    const double end = lip_window_end(from, direction);
    const int cells = static_cast<int>(std::lround(std::abs(end - from) / kLipTableStep));
    for (int k = 1; k <= cells; ++k) {
        const double along = std::copysign(k * kLipTableStep, direction);
        lowest = std::min(lowest, lip_limit_cached(from + along) +
                                      kLipEnvelopeSlope * std::abs(along));
    }
    return lowest;
}

double PouringController::tilt_rate_cap(double direction, double dt)
{
    // The tilt used to run at the law's rate until the IK fell behind and then
    // be held or braked -- seen after it happened, a delay that restarts from
    // rest every time. This is the rate it can be followed at, planned from
    // where it is: the ramp brakes into it smoothly as the path changes.
    tilt_cap_ik_ = std::numeric_limits<double>::infinity();
    tilt_cap_lip_ = std::numeric_limits<double>::infinity();
    const double need = joints_per_tilt(tilt_, inset_cmd_);
    if (need > 1e-9 && dt > 0.0) {
        tilt_cap_ik_ = kIkRateShare * (max_delta_q_ / dt) / need;
    }
    // The lip backs out at no more than max_lip_speed. Wherever ahead the
    // limit is closer in than the lip is now, the tilt may get there no sooner
    // than the lip can back out to it -- not tip, wait for the lip, tip again
    // (advance_tilt's hold_reason 2 and 3). The limit closes in steps, so this
    // looks over a stretch of the path, not at its slope here: the slope
    // between two steps is flat and at one is vertical, and a cap taken from
    // it came too late to brake for (2026-10-04, 0.53 then 0.10 rad/s).
    if (max_lip_speed_ > 0.0 && std::abs(direction) > 1e-12) {
        const double end = lip_window_end(tilt_, direction);
        const int cells = static_cast<int>(std::lround(std::abs(end - tilt_) / kLipTableStep));
        for (int k = 1; k <= cells; ++k) {
            const double along = std::copysign(k * kLipTableStep, direction);
            const double behind = inset_cmd_ - lip_limit_cached(tilt_ + along);
            if (behind > 1e-6) {
                tilt_cap_lip_ = std::min(tilt_cap_lip_,
                                         kLipRateShare * max_lip_speed_ * std::abs(along) / behind);
            }
        }
    }
    return std::min(tilt_cap_ik_, tilt_cap_lip_);
}

double PouringController::pour_joint_velocity_limit() const
{
    if (pour_index_ < 0 || pour_index_ >= model_.velocityLimit.size()) {
        return std::numeric_limits<double>::infinity();
    }
    const double v = model_.velocityLimit(pour_index_);
    return (std::isfinite(v) && v > 0.0) ? v : std::numeric_limits<double>::infinity();
}

double PouringController::pour_speed_limit(double dt) const
{
    // The per-cycle bound every command here already keeps to, and the joint's
    // own velocity limit from the description, whichever is lower. On the FR5
    // that is 0.625 rad/s against j6's 3.2: max_delta_q is the one that binds.
    const double per_cycle = dt > 0.0 ? max_delta_q_ / dt : 0.0;
    return std::min(per_cycle, pour_joint_velocity_limit());
}

bool PouringController::ik_blocked(double now, double factor) const
{
    return ik_blocked_since_ >= 0.0 && now - ik_blocked_since_ > factor * ik_fail_sec_;
}

std::string PouringController::blocked_reason() const
{
    const std::string pinned = joints_at_limits();
    std::ostringstream os;
    os << "the arm could not follow the lip path: the path was held for it at the edge of "
          "the IK tolerance for longer than ik_fail_sec and it did not catch up ("
       << (pinned.empty() ? "no joint at a limit: a singularity" : pinned) << ')';
    return os.str();
}

std::string PouringController::joints_at_limits() const
{
    constexpr double kNear = 0.01;
    std::ostringstream os;
    if (q_lower_limits_.size() != q_ref_.size()) {
        return "";
    }
    for (int i = 0; i < q_ref_.size(); ++i) {
        const bool low = q_ref_(i) <= q_lower_limits_(i) + kNear;
        const bool high = q_ref_(i) >= q_upper_limits_(i) - kNear;
        if (low || high) {
            os << (os.str().empty() ? "" : ", ")
               << (i < static_cast<int>(joint_names_.size()) ? joint_names_[i] : std::to_string(i))
               << " at its " << (low ? "lower" : "upper") << " limit";
        }
    }
    return os.str();
}

void PouringController::advance_tilt(double proposed, double dt)
{
    hold_reason_ = 0;
    // Nothing advances while the arm is far behind the path (kDeadHoldRatio):
    // the next target would only be further from where it is. That is the last
    // resort. tilt_rate_cap() keeps the tilt to what the IK can follow; past
    // the tolerance ik_speed_scale() is 0 and the ramp decelerates, and past
    // ik_hold_ratio the caller brakes it hard -- because holding it dead there
    // stopped the tilt within a cycle and restarted it from rest, every ~0.6 s
    // on the way back upright (2026-10-03, a 125 Hz trace of a syrup pour).
    if (ik_lagging_ && ik_lag_ratio_ >= kDeadHoldRatio) {
        hold_reason_ = 1;
        return;
    }
    const double goal = lip_path_.inset_limit(proposed);
    const double step = max_lip_speed_ * dt;
    // Where the lip heads: the limit a little way ahead in the direction the
    // tilt moves, so it is already out where the limit closes in a step and
    // the tilt never has to wait for it. Out further than it has to be is
    // further from the receiver.
    const double ahead = lip_ahead(proposed, proposed - tilt_);
    if (goal < inset_cmd_ - step) {
        // At the proposed tilt the lip has to be further out than it is, by
        // more than it can move in a cycle. Tipping the whole way first would
        // carry the jaw or the wall into the receiver while the lip caught up,
        // so the tilt takes only the share of its step the lip can keep pace
        // with. It used to take none of it and wait for the lip, which on the
        // way back upright alternated tip / back out / tip every cycle at the
        // boundary: a knock every half-second on the rig (2026-09-28).
        const double need = inset_cmd_ - goal;
        // 0.9: a share cut exactly to the boundary lands on either side of it
        // by rounding and the path's curvature, which is the alternation again.
        const double share = std::clamp(0.9 * step / need, 0.0, 1.0);
        const double partial = tilt_ + share * (proposed - tilt_);
        if (lip_path_.inset_limit(partial) >= inset_cmd_ - step) {
            tilt_ = partial;
            hold_reason_ = 3;
        } else {
            hold_reason_ = 2;
        }
        inset_cmd_ -= step;
    } else {
        // ahead <= goal, so wherever this lands the lip is inside the limit at
        // the proposed tilt.
        const double in_step =
            std::min(step, kLipEnvelopeSlope * std::abs(proposed - tilt_) + kLipCreep * dt);
        inset_cmd_ = std::clamp(ahead, inset_cmd_ - step, inset_cmd_ + in_step);
        tilt_ = proposed;
    }
    peak_tilt_ = std::max(peak_tilt_, std::abs(tilt_));
}

controller_interface::return_type PouringController::update_measured(double now, double dt)
{
    std::string why;

    if (phase_ == Phase::Measure) {
        hold_reference();
        last_phase_ = static_cast<std::uint8_t>(PourPhase::Verify);
        if (action_server_->is_canceling()) {
            abort_before_motion("the goal was cancelled before the pour began");
            return controller_interface::return_type::OK;
        }
        Eigen::VectorXd q_now = state_.q;
        q_now.head(num_dof_) = q_ref_;
        pinocchio::SE3 H_now;
        Eigen::MatrixXd J_unused;
        compute_arm_kinematics(q_now, H_now, J_unused);
        const Eigen::Isometry3d ee_now = to_isometry(H_now);

        if (!held_measured_) {
            const auto & bounds = action_server_->bounds();
            const bool measured_earlier = !bounds.grasp_joints.empty();
            VesselSample sample = *vessel_buffer_.readFromRT();
            // The vessel's axis as the kinematics know it: upright, as the
            // marker's pose assumes, unless the grasp was measured earlier and
            // the EE has turned it since.
            Eigen::Vector3d axis_now = Eigen::Vector3d::UnitZ();
            if (measured_earlier) {
                // The goal carries a grasp measured before it: the marker's
                // base-frame centre and the joints the arm was in. The grasp is a
                // rigid offset, so the marker's place in the EE frame then is its
                // place now -- carried to the present by this controller's own
                // kinematics, the same model the lip path is followed with.
                Eigen::VectorXd q_then = state_.q;
                for (int i = 0; i < num_dof_; ++i) {
                    q_then(i) = bounds.grasp_joints[i];
                }
                pinocchio::SE3 H_then;
                compute_arm_kinematics(q_then, H_then, J_unused);
                const Eigen::Vector3d marker_then(bounds.grasp_marker[0], bounds.grasp_marker[1],
                                                  bounds.grasp_marker[2]);
                // The vessel was upright when it was measured -- standing on the
                // bench in the jaws. Whatever the EE has turned through since, it
                // has turned the vessel through too.
                axis_now = H_now.rotation() * H_then.rotation().transpose() *
                           Eigen::Vector3d::UnitZ();
                const double lean = pour::lean_of(axis_now);
                // Levelling stands it up again, within its own bound, below.
                if (!level_enabled_ && lean > max_carry_lean_) {
                    std::ostringstream os;
                    os << "the pour was not started: the vessel was measured upright at its "
                       << "grasp, and the carry since has tipped it " << std::fixed
                       << std::setprecision(3) << lean << " rad (limit " << max_carry_lean_
                       << " rad). The lip path assumes it hangs upright when the pour starts";
                    abort_before_motion(os.str());
                    return controller_interface::return_type::OK;
                }
                sample.position = ee_now * (to_isometry(H_then).inverse() * marker_then);
                sample.finite = sample.position.allFinite();
                sample.frame_ok = true;
                sample.stamp = now;
                if (!sample.finite) {
                    abort_before_motion("the pour was not started: the grasp measured before "
                                        "the goal does not give a finite marker position");
                    return controller_interface::return_type::OK;
                }
            } else if (!vessel_sample_usable(sample, now)) {
                if (now - measure_started_ > vessel_pose_timeout_) {
                    std::ostringstream os;
                    os << "no usable pose of the held vessel within vessel_pose_timeout ("
                       << vessel_pose_timeout_ << " s): " << vessel_sample_problem(sample, now);
                    abort_before_motion(os.str());
                    return controller_interface::return_type::OK;
                }
                publish_feedback_if_due(dt);
                return controller_interface::return_type::OK;
            }
            // From here the vessel is rigid in the EE frame: where the jaws
            // hold it, which way its axis runs, how far below them its bottom
            // is. Without levelling, its bottom is the marker's point, exactly.
            held_ = pour::held_from_grasp(
                ee_now, sample.position, axis_now,
                lip_config_.grasp_depth * lip_config_.approach_axis_in_ee.normalized());
            held_measured_ = true;
        }

        if (level_enabled_) {
            // The tag's word on the axis, when there is a tag to ask: its pose
            // has to have been averaged with the arm standing still, like the
            // marker's.
            std::optional<Eigen::Vector3d> tag_axis;
            if (!attitude_topic_.empty()) {
                const AttitudeSample attitude = *attitude_buffer_.readFromRT();
                if (attitude_usable(attitude, now)) {
                    tag_axis = attitude.axis;
                } else if (now - measure_started_ <= vessel_pose_timeout_) {
                    publish_feedback_if_due(dt);
                    return controller_interface::return_type::OK;
                } else {
                    std::ostringstream os;
                    os << "no usable attitude of the vessel in the jaws on " << attitude_topic_
                       << " within vessel_pose_timeout (" << vessel_pose_timeout_ << " s): "
                       << (attitude.stamp <= 0.0
                               ? "nothing has arrived. Is the pose node publishing it for the "
                                 "vessel the robot has taken, and can a side camera see its tag?"
                               : "none arrived settled, fresh and in the base frame");
                    if (level_require_tag_) {
                        abort_before_motion("the pour was not started: " + os.str());
                        return controller_interface::return_type::OK;
                    }
                    RCLCPP_WARN(get_node()->get_logger(),
                        "%s; standing the vessel upright by the axis the grasp gave it instead",
                        os.str().c_str());
                }
            }
            const Eigen::Vector3d kinematic = held_.axis(ee_now);
            const pour::AxisChoice choice = pour::choose_axis(kinematic, tag_axis, level_config_);
            const double lean = pour::lean_of(choice.axis);
            std::ostringstream note;
            note << std::fixed << std::setprecision(1);
            if (tag_axis && !choice.from_tag) {
                RCLCPP_WARN(get_node()->get_logger(),
                    "The tag puts the vessel's axis %.3f rad from where the grasp left it "
                    "(level.max_tag_disagreement %.3f rad): more likely a flipped tag pose than "
                    "a vessel turned that far in the jaws. Standing it upright by the grasp's",
                    choice.disagreement, level_config_.max_tag_disagreement);
            }
            note << "leaning " << lean * 180.0 / M_PI << " deg by its "
                 << (choice.from_tag ? "tag" : "grasp");
            if (choice.from_tag) {
                note << " (" << choice.disagreement * 180.0 / M_PI << " deg from the grasp's)";
            }
            if (lean > level_config_.max_correction) {
                std::ostringstream os;
                os << "the pour was not started: the vessel in the jaws is " << note.str()
                   << ", past level.max_correction (" << std::fixed << std::setprecision(1)
                   << level_config_.max_correction * 180.0 / M_PI
                   << " deg). Check how it hangs before pouring from it";
                abort_before_motion(os.str());
                return controller_interface::return_type::OK;
            }
            // The jaws still hold it where they did; it is its axis that is now
            // the one chosen.
            held_.axis_in_ee = H_now.rotation().transpose() * choice.axis;
            std::string why;
            if (!level_move_.plan(ee_now, choice.axis, held_.grip(ee_now), why)) {
                abort_before_motion("the pour was not started: " + why);
                return controller_interface::return_type::OK;
            }
            if (lean > level_config_.deadband) {
                note << ", stood upright about the jaws";
                level_note_ = note.str();
                level_s_ = 0.0;
                level_ramp_.reset();
                phase_ = Phase::Level;
                RCLCPP_INFO(get_node()->get_logger(),
                    "Levelling: the vessel is %s; turning the EE %.3f rad about the jaws at "
                    "(%.4f, %.4f, %.4f) before the pour", level_note_.c_str(), level_move_.angle(),
                    level_move_.pivot().x(), level_move_.pivot().y(), level_move_.pivot().z());
                publish_feedback_if_due(dt);
                return controller_interface::return_type::OK;
            }
            note << ", within level.deadband: not turned";
            level_note_ = note.str();
        }

        std::string why;
        if (!plan_pour_geometry(why)) {
            abort_before_motion("the pour was not started: " + why);
            return controller_interface::return_type::OK;
        }
        align_s_ = 0.0;
        phase_ = Phase::Align;
    }

    if (phase_ == Phase::Level) {
        last_phase_ = static_cast<std::uint8_t>(PourPhase::Verify);
        std::string why;
        if (action_server_->is_canceling()) {
            level_ramp_.reset();
            fail_geometric("the goal was cancelled before the pour began", Phase::Unlevel);
        } else if (level_s_ >= level_move_.angle() && !ik_lagging_) {
            if (!plan_pour_geometry(why)) {
                level_ramp_.reset();
                fail_geometric("the pour was not started: " + why, Phase::Unlevel);
            } else {
                align_s_ = 0.0;
                align_ramp_.reset();
                phase_ = Phase::Align;
            }
        } else {
            if (!ik_lagging_) {
                level_s_ += level_ramp_.reach(level_s_, level_move_.angle(),
                                              level_rate_ * ik_speed_scale(), tilt_accel_, dt);
            } else if (ik_lag_ratio_ < kDeadHoldRatio) {
                level_s_ = std::clamp(level_s_ + level_ramp_.brake(kEmergencyBrake * tilt_accel_, dt),
                                      0.0, level_move_.angle());
            } else {
                level_ramp_.reset();
            }
            if (!track(level_move_.at(level_s_), now, why)) {
                fail_geometric(why, Phase::Return);
            } else if (ik_blocked(now)) {
                fail_geometric(blocked_reason(), Phase::Return);
            }
            publish_feedback_if_due(dt);
            return controller_interface::return_type::OK;
        }
    }

    if (phase_ == Phase::Align) {
        last_phase_ = static_cast<std::uint8_t>(PourPhase::Verify);
        if (action_server_->is_canceling()) {
            fail_geometric("the goal was cancelled before the pour began", Phase::Unalign);
        } else if (align_s_ >= lip_path_.align_length() && !ik_lagging_) {
            inset_cmd_ = lip_path_.inset_limit(0.0);
            start_pour(now);
        } else {
            if (!ik_lagging_) {
                align_s_ += align_ramp_.reach(align_s_, lip_path_.align_length(),
                                              align_speed_ * ik_speed_scale(), align_accel_, dt);
            } else if (ik_lag_ratio_ < kDeadHoldRatio) {
                align_s_ = std::clamp(align_s_ + align_ramp_.brake(kEmergencyBrake * align_accel_, dt),
                                      0.0, lip_path_.align_length());
            } else {
                align_ramp_.reset();
            }
            if (!track(lip_path_.ee_pose_aligning(align_s_), now, why)) {
                fail_geometric(why, Phase::Return);
            } else if (ik_blocked(now)) {
                fail_geometric(blocked_reason(), Phase::Return);
            }
            publish_feedback_if_due(dt);
            return controller_interface::return_type::OK;
        }
    }

    if (phase_ == Phase::Pouring || phase_ == Phase::Untilt) {
        const double tilt_before = tilt_;
        double proposed = tilt_;
        // Blocked on the way UP is the arm's reach from this pose, not a
        // failure: the tilt stops there and the law is told, so its own rules
        // for a vessel at its bound take over -- the stall, the taps, the park.
        // Without this a seek sat at 77.6 deg for 140 s (2026-09-28), j4 0.6
        // deg from its limit, until the goal was cancelled by hand.
        if (phase_ == Phase::Pouring && last_law_rate_ > 0.0 && ik_blocked(now) &&
            tilt_ < reach_) {
            reach_ = tilt_;
            tilt_bound_ = std::min(tilt_bound_, reach_);
            ik_blocked_since_ = -1.0;
            const std::string pinned = joints_at_limits();
            RCLCPP_WARN(get_node()->get_logger(),
                "The arm reaches no further than %.3f rad (%.1f deg) of tilt from this pose (%s); "
                "the pour takes that as its bound", reach_, reach_ * 180.0 / M_PI,
                pinned.empty() ? "no joint at a limit: a singularity" : pinned.c_str());
        }
        // Well behind the path the tilt is braked hard rather than held: the
        // law still runs, but what it asks waits until the arm has caught up.
        const bool braking = ik_lagging_ && ik_lag_ratio_ >= ik_hold_ratio_ &&
                             ik_lag_ratio_ < kDeadHoldRatio;
        if (phase_ == Phase::Pouring) {
            const double rate = step_law(now);
            if (phase_ == Phase::Pouring) {
                const double capped =
                    std::copysign(std::min(std::abs(rate) * ik_speed_scale(),
                                           tilt_rate_cap(std::isfinite(law_target_tilt_)
                                                             ? law_target_tilt_ - tilt_
                                                             : rate,
                                                         dt)),
                                  rate);
                const double step = braking ? tilt_ramp_.brake(kEmergencyBrake * tilt_accel_, dt)
                                            : shaped_law_step(capped, dt);
                proposed = std::clamp(tilt_ + step, -max_back_tilt_, tilt_bound_);
            }
        }
        if (phase_ == Phase::Untilt) {
            last_phase_ = static_cast<std::uint8_t>(PourPhase::Done);
            const double rate = std::min({law_->return_tilt_rate(),
                                          max_delta_q_ / std::max(dt, 1e-9),
                                          tilt_rate_cap(-tilt_, dt)});
            proposed = tilt_ + (braking ? tilt_ramp_.brake(kEmergencyBrake * tilt_accel_, dt)
                                        : tilt_ramp_.reach(tilt_, 0.0, rate * ik_speed_scale(),
                                                           tilt_accel_, dt));
        }
        advance_tilt(proposed, dt);
        if (braking && hold_reason_ == 0) {
            hold_reason_ = 4;
        }
        // advance_tilt holds the tilt back while the IK lags or the lip backs
        // out; the ramp carries on from what it actually did.
        tilt_ramp_.observed(tilt_ - tilt_before, dt);
        publish_tilt_debug(now, proposed);

        const bool home = phase_ == Phase::Untilt && tilt_ == 0.0 && !ik_lagging_ &&
                          inset_cmd_ >= lip_path_.inset_limit(0.0) - 1e-9;
        if (home) {
            align_s_ = lip_path_.align_length();
            align_ramp_.reset();
            phase_ = Phase::Unalign;
        } else {
            if (!track(lip_path_.ee_pose(tilt_, inset_cmd_), now, why)) {
                fail_geometric(why, Phase::Return);
            } else if (phase_ == Phase::Untilt && ik_blocked(now, kReturnBlockFactor)) {
                // On the way back the path retraces the way in, so a block
                // here is not a reach: it is the old failure to follow.
                fail_geometric(blocked_reason(), Phase::Return);
            }
            publish_feedback_if_due(dt);
            return controller_interface::return_type::OK;
        }
    }

    if (phase_ == Phase::Unalign) {
        last_phase_ = static_cast<std::uint8_t>(PourPhase::Done);
        if (align_s_ <= 0.0 && !ik_lagging_) {
            level_ramp_.reset();
            phase_ = level_s_ > 0.0 ? Phase::Unlevel : Phase::Return;
        } else {
            if (!ik_lagging_) {
                align_s_ += align_ramp_.reach(align_s_, 0.0, align_speed_ * ik_speed_scale(),
                                              align_accel_, dt);
            } else if (ik_lag_ratio_ < kDeadHoldRatio) {
                align_s_ = std::clamp(align_s_ + align_ramp_.brake(kEmergencyBrake * align_accel_, dt),
                                      0.0, lip_path_.align_length());
            } else {
                align_ramp_.reset();
            }
            if (!track(lip_path_.ee_pose_aligning(align_s_), now, why)) {
                fail_geometric(why, Phase::Return);
            } else if (ik_blocked(now, kReturnBlockFactor)) {
                fail_geometric(blocked_reason(), Phase::Return);
            }
            publish_feedback_if_due(dt);
            return controller_interface::return_type::OK;
        }
    }

    if (phase_ == Phase::Unlevel) {
        // Back the way Level came, as slowly: the vessel may still be full.
        last_phase_ = static_cast<std::uint8_t>(PourPhase::Done);
        std::string why;
        if (level_s_ <= 0.0 && !ik_lagging_) {
            phase_ = Phase::Return;
        } else {
            if (!ik_lagging_) {
                level_s_ += level_ramp_.reach(level_s_, 0.0, level_rate_ * ik_speed_scale(),
                                              tilt_accel_, dt);
            } else if (ik_lag_ratio_ < kDeadHoldRatio) {
                level_s_ = std::clamp(level_s_ + level_ramp_.brake(kEmergencyBrake * tilt_accel_, dt),
                                      0.0, level_move_.angle());
            } else {
                level_ramp_.reset();
            }
            if (!track(level_move_.at(level_s_), now, why)) {
                fail_geometric(why, Phase::Return);
            } else if (ik_blocked(now, kReturnBlockFactor)) {
                fail_geometric(blocked_reason(), Phase::Return);
            }
            publish_feedback_if_due(dt);
            return controller_interface::return_type::OK;
        }
    }

    if (phase_ == Phase::Return) {
        // Joint space, to the configuration the goal started in. After a clean
        // unalign this is the IK's residual, a fraction of a milliradian; after
        // a failure to follow the path it is the whole way back.
        last_phase_ = static_cast<std::uint8_t>(PourPhase::Done);
        Eigen::VectorXd delta = q_start_ - q_ref_;
        if (delta.cwiseAbs().maxCoeff() <= 1e-9) {
            write_command(q_start_);
            finish();
            return controller_interface::return_type::OK;
        }
        delta = delta.array().max(-max_delta_q_).min(max_delta_q_);
        write_command(q_ref_ + delta);
        publish_feedback_if_due(dt);
        return controller_interface::return_type::OK;
    }

    hold_reference();
    return controller_interface::return_type::OK;
}

} // namespace sdl_pour_controller

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(sdl_pour_controller::PouringController,
                       controller_interface::ControllerInterface)
