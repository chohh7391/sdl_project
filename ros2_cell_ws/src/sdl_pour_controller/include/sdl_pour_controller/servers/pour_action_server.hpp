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

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "sdl_pour_controller/servers/base_action_server.hpp"
#include "sdl_cell_interfaces/action/pour.hpp"

namespace sdl_pour_controller {

using PourAction = sdl_cell_interfaces::action::Pour;
using PourGoalHandle = rclcpp_action::ServerGoalHandle<PourAction>;

//: The goal, with every unset (zero) BOUND already replaced by the controller's
//: configured default. Resolved once when the goal is accepted so the control
//: loop never re-reads the goal message.
//:
//: container_grams, material and flow_index are not bounds and get no defaults:
//: they describe what is being poured and into what, and a controller has no
//: business guessing either.
struct PourBounds {
    double target_grams{0.0};
    double container_grams{0.0};
    std::uint8_t material{0};
    double flow_index{0.0};
    double max_tilt_rate{0.0};
    double max_tilt{0.0};
    double tolerance{0.0};
    double timeout{0.0};
    //: A grasp measured before the goal: the joints it was measured in and the
    //: marker's base-frame centre then. Empty joints: measure at goal start.
    std::vector<double> grasp_joints;
    std::array<double, 3> grasp_marker{{0.0, 0.0, 0.0}};
    //: +1 / -1 from the goal, or 0 for the controller's configured one.
    int pour_direction{0};
    //: A configuration showing how the pour tips the vessel; empty: the pour
    //: joint's axis.
    std::vector<double> pour_reference_joints;
    //: The receiver's rim centre in the base frame, from the goal: three
    //: numbers, or empty where the controller does not need one.
    std::vector<double> receiver_rim_center;
};

/**
 * Goal lifecycle for a pour. The CONTROL LAW is not here.
 *
 * Every server in cho_controller_fr5 owns a trajectory and the controller
 * samples it. A pour has no trajectory to sample: where the wrist goes next is
 * decided by a weight that arrives while the goal runs, so the decisions live in
 * sdl_pour_controller/pour/ and the wiring in PouringController, and this owns
 * only what a goal is: validating it, holding it, reporting feedback, and ending
 * it exactly once.
 */
class FR5PourActionServer : public FR5BaseActionServer<PourAction>
{
public:
    using FR5BaseActionServer<PourAction>::FR5BaseActionServer;

    rclcpp_action::GoalResponse handle_goal(
        const rclcpp_action::GoalUUID & uuid,
        std::shared_ptr<const PourAction::Goal> goal) override;

    rclcpp_action::CancelResponse handle_cancel(
        const std::shared_ptr<PourGoalHandle> goal_handle) override;

    void handle_accepted(const std::shared_ptr<PourGoalHandle> goal_handle) override;

    // Nothing to advance per cycle beyond the cancel check: the controller runs
    // the law and calls succeed()/abort() itself.
    bool compute(const rclcpp::Time & current_time, cho_controller::fr5::FR5State & state) override;

    // Defaults the controller hands over, used to fill a goal's zero bounds.
    void set_defaults(const PourBounds & defaults) { defaults_ = defaults; }
    // The tilt bound a granular goal gets when it sets none; 0: the defaults'.
    void set_granular_max_tilt(double max_tilt) { granular_max_tilt_ = max_tilt; }
    // The timeout a granular goal gets when it sets none; 0: the defaults'.
    void set_granular_timeout(double timeout) { granular_timeout_ = timeout; }
    // Whether a goal must say where the receiver is (pour_geometry: measured).
    void set_requires_receiver(bool required) { requires_receiver_ = required; }
    const PourBounds & bounds() const { return bounds_; }

    bool is_canceling() const;
    void publish_feedback(double grams, double tilt, double elapsed, double flow_rate,
                          std::uint8_t phase);
    bool succeed(double grams, double peak_tilt, double elapsed, int trim_pulses,
                 double measured_afterflow);
    bool abort_active_goal(const std::string & reason, double grams = 0.0,
                           double peak_tilt = 0.0, double elapsed = 0.0,
                           int trim_pulses = 0, double measured_afterflow = 0.0);

private:
    bool requires_receiver_{false};
    double granular_max_tilt_{0.0};
    double granular_timeout_{0.0};
    PourBounds defaults_;
    PourBounds bounds_;
};

} // namespace sdl_pour_controller
