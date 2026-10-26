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

#include "sdl_pour_controller/pour/level.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace sdl_pour_controller {
namespace pour {

namespace {
//: `axis` normalised and turned to point up: an axis, not a direction.
Eigen::Vector3d upward(const Eigen::Vector3d & axis)
{
    const Eigen::Vector3d unit = axis.normalized();
    return unit.z() < 0.0 ? Eigen::Vector3d(-unit) : unit;
}

double angle_between(const Eigen::Vector3d & a, const Eigen::Vector3d & b)
{
    // atan2 of |a x b| and a . b stays accurate near 0, where acos does not.
    return std::atan2(a.cross(b).norm(), a.dot(b));
}
}  // namespace

double lean_of(const Eigen::Vector3d & axis)
{
    return angle_between(axis.normalized(), Eigen::Vector3d::UnitZ());
}

HeldInJaws held_from_grasp(const Eigen::Isometry3d & ee, const Eigen::Vector3d & bottom,
                           const Eigen::Vector3d & axis, const Eigen::Vector3d & jaws_in_ee)
{
    const Eigen::Vector3d a = axis.normalized();
    const double height = (ee * jaws_in_ee - bottom).dot(a);
    HeldInJaws held;
    held.grip_in_ee = ee.inverse() * Eigen::Vector3d(bottom + height * a);
    held.axis_in_ee = ee.linear().transpose() * a;
    held.grip_height = height;
    return held;
}

bool LevelConfig::validate(std::string & why) const
{
    const std::pair<const char *, double> angles[] = {
        {"max_correction", max_correction},
        {"max_tag_disagreement", max_tag_disagreement},
        {"deadband", deadband},
    };
    for (const auto & [name, value] : angles) {
        // Past half a right angle none of these is a small correction any more.
        if (!std::isfinite(value) || value <= 0.0 || value > M_PI_4) {
            std::ostringstream os;
            os << "level." << name << " must be an angle in (0, pi/4] rad (got " << value << ')';
            why = os.str();
            return false;
        }
    }
    if (deadband >= max_correction) {
        why = "level.deadband must be under level.max_correction: it is the lean not worth "
              "moving for, and max_correction the most that is corrected";
        return false;
    }
    return true;
}

AxisChoice choose_axis(const Eigen::Vector3d & kinematic, const std::optional<Eigen::Vector3d> & tag,
                       const LevelConfig & config)
{
    AxisChoice choice;
    choice.axis = upward(kinematic);
    if (!tag || !tag->allFinite() || tag->norm() < 1e-9) {
        return choice;
    }
    const Eigen::Vector3d seen = upward(*tag);
    choice.disagreement = angle_between(seen, choice.axis);
    if (choice.disagreement <= config.max_tag_disagreement) {
        choice.axis = seen;
        choice.from_tag = true;
    }
    return choice;
}

bool LevelMove::plan(const Eigen::Isometry3d & ee, const Eigen::Vector3d & axis,
                     const Eigen::Vector3d & pivot, std::string & why)
{
    angle_ = 0.0;
    start_ = ee;
    pivot_ = pivot;
    if (!ee.matrix().allFinite() || !axis.allFinite() || !pivot.allFinite() || axis.norm() < 1e-9) {
        why = "non-finite EE pose, vessel axis or pivot";
        return false;
    }
    const Eigen::Vector3d a = axis.normalized();
    if (a.z() <= 0.0) {
        why = "the vessel's axis points level or down: it is not hanging from the jaws";
        return false;
    }
    const Eigen::Vector3d turn = a.cross(Eigen::Vector3d::UnitZ());
    angle_ = std::atan2(turn.norm(), a.z());
    rotation_axis_ = turn.norm() > 1e-12 ? Eigen::Vector3d(turn.normalized())
                                         : Eigen::Vector3d::UnitX();
    return true;
}

Eigen::Isometry3d LevelMove::at(double s) const
{
    const Eigen::Matrix3d R =
        Eigen::AngleAxisd(std::clamp(s, 0.0, angle_), rotation_axis_).toRotationMatrix();
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() = R * start_.linear();
    T.translation() = pivot_ + R * (start_.translation() - pivot_);
    return T;
}

} // namespace pour
} // namespace sdl_pour_controller
