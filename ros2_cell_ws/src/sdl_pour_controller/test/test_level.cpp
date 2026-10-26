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

// Standing the held vessel upright before the pour, against a synthetic side
// grasp. No robot: the EE pose is constructed.

#include <gtest/gtest.h>

#include <cmath>
#include <optional>
#include <string>

#include "sdl_pour_controller/pour/level.hpp"

using sdl_pour_controller::pour::AxisChoice;
using sdl_pour_controller::pour::choose_axis;
using sdl_pour_controller::pour::held_from_grasp;
using sdl_pour_controller::pour::HeldInJaws;
using sdl_pour_controller::pour::lean_of;
using sdl_pour_controller::pour::LevelConfig;
using sdl_pour_controller::pour::LevelMove;

namespace {

constexpr double kDeg = M_PI / 180.0;
constexpr double kGraspDepth = 0.285;   // wrist3_link to the jaws, on the 15 mm flange
constexpr double kGripHeight = 0.046;   // the jaws above the beaker's bottom, as logged

// The FR5's side grasp: the jaws close along EE x (base x here), the approach
// is EE z (base +y, level), so EE y is base -z.
Eigen::Isometry3d side_grasp()
{
    Eigen::Isometry3d ee = Eigen::Isometry3d::Identity();
    ee.linear().col(0) = Eigen::Vector3d::UnitX();
    ee.linear().col(1) = -Eigen::Vector3d::UnitZ();
    ee.linear().col(2) = Eigen::Vector3d::UnitY();
    ee.translation() = Eigen::Vector3d(0.45, 0.32 - kGraspDepth, 0.18);
    return ee;
}

const Eigen::Vector3d kJawsInEe{0.0, 0.0, kGraspDepth};

// The beaker as it stood when the jaws closed: upright, its axis through them.
HeldInJaws upright_grasp(const Eigen::Isometry3d & ee)
{
    const Eigen::Vector3d jaws = ee * kJawsInEe;
    return held_from_grasp(ee, jaws - kGripHeight * Eigen::Vector3d::UnitZ(),
                           Eigen::Vector3d::UnitZ(), kJawsInEe);
}

// The beaker turned `angle` about the jaws' closing axis, through the grip.
Eigen::Vector3d turned_in_jaws(double angle)
{
    return Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitX()) * Eigen::Vector3d::UnitZ();
}

}  // namespace

TEST(Level, TheGraspPutsTheGripOnTheAxisAtTheJawsHeight)
{
    const Eigen::Isometry3d ee = side_grasp();
    const HeldInJaws held = upright_grasp(ee);
    EXPECT_NEAR(held.grip_height, kGripHeight, 1e-12);
    EXPECT_TRUE(held.grip(ee).isApprox(ee * kJawsInEe, 1e-12));
    EXPECT_TRUE(held.bottom(ee).isApprox(
        Eigen::Vector3d(0.45, 0.32, 0.18 - kGripHeight), 1e-12));
    EXPECT_NEAR(lean_of(held.axis(ee)), 0.0, 1e-12);
}

TEST(Level, AMarkerOffTheJawsLineStillPutsTheGripOnTheVesselsAxis)
{
    // Perception put the beaker 4 mm along the jaws' closing axis from where
    // they closed: the grip is on the beaker's axis, at the jaws' height.
    const Eigen::Isometry3d ee = side_grasp();
    const Eigen::Vector3d bottom = ee * kJawsInEe + Eigen::Vector3d(0.004, 0.0, -kGripHeight);
    const HeldInJaws held = held_from_grasp(ee, bottom, Eigen::Vector3d::UnitZ(), kJawsInEe);
    EXPECT_TRUE(held.bottom(ee).isApprox(bottom, 1e-12));
    EXPECT_NEAR(held.grip(ee).x(), bottom.x(), 1e-12);
    EXPECT_NEAR(held.grip_height, kGripHeight, 1e-12);
}

TEST(Level, AnUprightVesselNeedsNoTurn)
{
    const Eigen::Isometry3d ee = side_grasp();
    const HeldInJaws held = upright_grasp(ee);
    LevelMove move;
    std::string why;
    ASSERT_TRUE(move.plan(ee, held.axis(ee), held.grip(ee), why)) << why;
    EXPECT_NEAR(move.angle(), 0.0, 1e-12);
    EXPECT_TRUE(move.at(0.1).isApprox(ee, 1e-12));
}

TEST(Level, TheTagIsBelievedWhenItAgreesWithTheKinematics)
{
    const LevelConfig config;
    const AxisChoice choice = choose_axis(Eigen::Vector3d::UnitZ(), turned_in_jaws(5.0 * kDeg),
                                          config);
    EXPECT_TRUE(choice.from_tag);
    EXPECT_NEAR(choice.disagreement, 5.0 * kDeg, 1e-9);
    EXPECT_NEAR(lean_of(choice.axis), 5.0 * kDeg, 1e-9);
}

TEST(Level, ATagThatDisagreesFarIsAFlippedPoseNotATurnedVessel)
{
    const LevelConfig config;
    const AxisChoice choice = choose_axis(Eigen::Vector3d::UnitZ(), turned_in_jaws(20.0 * kDeg),
                                          config);
    EXPECT_FALSE(choice.from_tag);
    EXPECT_NEAR(choice.disagreement, 20.0 * kDeg, 1e-9);
    EXPECT_NEAR(lean_of(choice.axis), 0.0, 1e-12);
}

TEST(Level, ATagNormalPointingDownIsTheSameAxis)
{
    const LevelConfig config;
    const AxisChoice choice = choose_axis(Eigen::Vector3d::UnitZ(), -turned_in_jaws(3.0 * kDeg),
                                          config);
    EXPECT_TRUE(choice.from_tag);
    EXPECT_GT(choice.axis.z(), 0.0);
    EXPECT_NEAR(lean_of(choice.axis), 3.0 * kDeg, 1e-9);
}

TEST(Level, WithoutATagTheKinematicsDecide)
{
    const LevelConfig config;
    const Eigen::Vector3d carried = turned_in_jaws(4.0 * kDeg);
    const AxisChoice choice = choose_axis(carried, std::nullopt, config);
    EXPECT_FALSE(choice.from_tag);
    EXPECT_TRUE(std::isnan(choice.disagreement));
    EXPECT_TRUE(choice.axis.isApprox(carried, 1e-12));
}

TEST(Level, TurningAboutTheGripStandsAVesselThatTurnedInTheJawsUpright)
{
    const Eigen::Isometry3d ee = side_grasp();
    HeldInJaws held = upright_grasp(ee);
    // It turned 6 degrees about the closing axis, through the grip, after the
    // grasp: the tag's axis replaces the one the grasp gave it.
    const Eigen::Vector3d seen = turned_in_jaws(6.0 * kDeg);
    held.axis_in_ee = ee.linear().transpose() * seen;
    const Eigen::Vector3d grip_before = held.grip(ee);

    LevelMove move;
    std::string why;
    ASSERT_TRUE(move.plan(ee, held.axis(ee), grip_before, why)) << why;
    EXPECT_NEAR(move.angle(), 6.0 * kDeg, 1e-9);
    const Eigen::Isometry3d levelled = move.at(move.angle());

    EXPECT_NEAR(lean_of(held.axis(levelled)), 0.0, 1e-9);
    // The jaws stay where they hold the vessel; the bottom comes back under them.
    EXPECT_TRUE(held.grip(levelled).isApprox(grip_before, 1e-12));
    EXPECT_TRUE(held.bottom(levelled).isApprox(
        grip_before - kGripHeight * Eigen::Vector3d::UnitZ(), 1e-9));
    // The wrist swings on the pivot: its distance from the grip does not change.
    EXPECT_NEAR((levelled.translation() - grip_before).norm(),
                (ee.translation() - grip_before).norm(), 1e-12);
}

TEST(Level, TheTurnIsGradedAndClampedAtItsEnds)
{
    const Eigen::Isometry3d ee = side_grasp();
    const Eigen::Vector3d axis = turned_in_jaws(8.0 * kDeg);
    LevelMove move;
    std::string why;
    ASSERT_TRUE(move.plan(ee, axis, ee * kJawsInEe, why)) << why;
    const auto lean_at = [&](double s) {
        return lean_of(move.at(s).linear() * (ee.linear().transpose() * axis));
    };
    EXPECT_NEAR(lean_at(0.0), 8.0 * kDeg, 1e-9);
    EXPECT_NEAR(lean_at(3.0 * kDeg), 5.0 * kDeg, 1e-9);
    EXPECT_NEAR(lean_at(-1.0), 8.0 * kDeg, 1e-9);
    EXPECT_NEAR(lean_at(1.0), 0.0, 1e-9);
}

TEST(Level, AVesselNotHangingFromTheJawsIsRefused)
{
    const Eigen::Isometry3d ee = side_grasp();
    LevelMove move;
    std::string why;
    EXPECT_FALSE(move.plan(ee, Eigen::Vector3d::UnitX(), ee * kJawsInEe, why));
    EXPECT_NE(why.find("level or down"), std::string::npos);
    EXPECT_FALSE(move.plan(ee, -Eigen::Vector3d::UnitZ(), ee * kJawsInEe, why));
}

TEST(Level, TheConfigurationIsChecked)
{
    std::string why;
    EXPECT_TRUE(LevelConfig{}.validate(why)) << why;
    LevelConfig config;
    config.deadband = config.max_correction;
    EXPECT_FALSE(config.validate(why));
    config = LevelConfig{};
    config.max_correction = 1.0;
    EXPECT_FALSE(config.validate(why));
    config = LevelConfig{};
    config.max_tag_disagreement = std::nan("");
    EXPECT_FALSE(config.validate(why));
}
