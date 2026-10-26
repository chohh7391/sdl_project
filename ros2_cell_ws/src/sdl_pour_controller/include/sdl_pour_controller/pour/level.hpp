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

#pragma once

#include <limits>
#include <optional>
#include <string>

#include <Eigen/Geometry>

namespace sdl_pour_controller {
namespace pour {

/**
 * Standing the held vessel upright before the pour starts.
 *
 * The lip path (lip_path.hpp) assumes the vessel hangs upright at the attitude
 * the pour starts from. Two things know how it actually hangs:
 *
 *   the KINEMATICS  it stood upright on the bench when the jaws closed, so its
 *                   axis in the EE frame is the base z seen from the EE then,
 *                   and the EE has turned it through whatever the EE has turned
 *                   through since. Blind to a vessel that turned in the jaws.
 *   its TAG         beaker1's tag lies flat on the mount stuck under the
 *                   beaker and comes with it, so the tag's normal is the
 *                   beaker's axis. Sees a turn in the jaws, and brings a
 *                   camera's orientation error with it.
 *
 * The tag is believed unless it disagrees with the kinematics by more than
 * `max_tag_disagreement`: a vessel that turned that far in the jaws is less
 * likely than a tag pose that flipped. Then the vessel is turned upright about
 * the point where the jaws hold it, which is the one point of it the jaws keep
 * where it is, so the move is a rotation of the whole EE about that point.
 *
 * No ROS, no Pinocchio: poses in, poses out.
 */

//: The held vessel, rigid in the EE frame.
struct HeldInJaws {
    //: The point on the vessel's axis at the height the jaws hold it, EE frame.
    Eigen::Vector3d grip_in_ee{Eigen::Vector3d::Zero()};
    //: The vessel's axis, bottom to rim, unit, EE frame.
    Eigen::Vector3d axis_in_ee{Eigen::Vector3d::UnitZ()};
    //: Bottom to grip along the axis [m].
    double grip_height{0.0};

    [[nodiscard]] Eigen::Vector3d grip(const Eigen::Isometry3d & ee) const { return ee * grip_in_ee; }
    [[nodiscard]] Eigen::Vector3d axis(const Eigen::Isometry3d & ee) const
    {
        return ee.linear() * axis_in_ee;
    }
    //: The centre of the vessel's bottom, base frame, with the EE at `ee`.
    [[nodiscard]] Eigen::Vector3d bottom(const Eigen::Isometry3d & ee) const
    {
        return grip(ee) - grip_height * axis(ee);
    }
};

//: How far `axis` leans from the base z [rad], 0 for upright.
[[nodiscard]] double lean_of(const Eigen::Vector3d & axis);

//: The held vessel as the grasp measured it: its bottom centre `bottom` and its
//: axis `axis` in the base frame with the EE at `ee`, and the jaws' own point
//: `jaws_in_ee` (the EE-frame point the jaws hold at). The grip is where the
//: vessel's axis passes the jaws' height -- the jaws centre what they hold, so
//: the axis runs through them.
[[nodiscard]] HeldInJaws held_from_grasp(const Eigen::Isometry3d & ee, const Eigen::Vector3d & bottom,
                                         const Eigen::Vector3d & axis,
                                         const Eigen::Vector3d & jaws_in_ee);

struct LevelConfig {
    //: A vessel leaning further than this is not turned upright: the pour is
    //: refused before anything moves [rad].
    double max_correction{0.26};
    //: The tag's axis is believed only this close to the kinematics' [rad].
    double max_tag_disagreement{0.17};
    //: A lean under this is not worth moving for [rad].
    double deadband{0.01};

    bool validate(std::string & why) const;
};

//: Which axis the vessel is taken to have, and why.
struct AxisChoice {
    Eigen::Vector3d axis{Eigen::Vector3d::UnitZ()};
    bool from_tag{false};
    //: Angle between the tag's axis and the kinematics' [rad]; NaN without a tag.
    double disagreement{std::numeric_limits<double>::quiet_NaN()};
};

//: The tag's axis when there is one and it agrees with the kinematics to
//: within max_tag_disagreement, else the kinematics'. Both are taken as axes:
//: a tag normal pointing down is the same axis pointing up.
[[nodiscard]] AxisChoice choose_axis(const Eigen::Vector3d & kinematic,
                                     const std::optional<Eigen::Vector3d> & tag,
                                     const LevelConfig & config);

//: The EE rotating about a fixed point until the vessel's axis is upright.
class LevelMove
{
public:
    //: Plan from the EE at `ee`, the vessel's axis `axis` (base frame) and the
    //: point it turns about. False, with `why`, for a vessel upside down.
    bool plan(const Eigen::Isometry3d & ee, const Eigen::Vector3d & axis,
              const Eigen::Vector3d & pivot, std::string & why);

    //: The whole turn [rad], >= 0.
    [[nodiscard]] double angle() const { return angle_; }
    //: The EE pose `s` rad into the turn, clamped to [0, angle].
    [[nodiscard]] Eigen::Isometry3d at(double s) const;
    [[nodiscard]] const Eigen::Vector3d & rotation_axis() const { return rotation_axis_; }
    [[nodiscard]] const Eigen::Vector3d & pivot() const { return pivot_; }

private:
    Eigen::Isometry3d start_{Eigen::Isometry3d::Identity()};
    Eigen::Vector3d pivot_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d rotation_axis_{Eigen::Vector3d::UnitX()};
    double angle_{0.0};
};

} // namespace pour
} // namespace sdl_pour_controller
