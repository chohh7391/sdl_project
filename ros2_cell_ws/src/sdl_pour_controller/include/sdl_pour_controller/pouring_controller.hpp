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
#include <string>
#include <vector>

#include <realtime_tools/realtime_buffer.hpp>
#include <realtime_tools/realtime_publisher.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>

#include "cho_controller_fr5/base_controller.hpp"
#include "sdl_pour_controller/pour/level.hpp"
#include "sdl_pour_controller/pour/lip_path.hpp"
#include "sdl_pour_controller/pour/material_profile.hpp"
#include "sdl_pour_controller/pour/pour_planner.hpp"
#include "sdl_pour_controller/pour/ramp.hpp"
#include "sdl_pour_controller/pour/shaker.hpp"
#include "sdl_pour_controller/pour/shaping_law.hpp"
#include "sdl_pour_controller/pour/scale_filter.hpp"
#include "sdl_pour_controller/servers/pour_action_server.hpp"
#include "sdl_cell_interfaces/msg/scale_reading.hpp"

namespace sdl_pour_controller {

/**
 * Tilt a held vessel until a scale says enough has come out, then set it back.
 *
 * The other FR5 controllers follow a path someone else decided on. This one has
 * no path: how far the wrist turns next depends on a weight that arrives while
 * the goal runs, which is what makes the pour adaptive and what makes it a
 * controller rather than a trajectory.
 *
 * WHAT IS HERE AND WHAT IS NOT. The decisions -- when to stop, how large a trim
 * pulse is, whether the reading can be believed -- live in `pour/`, which has no
 * ROS in it at all: time is a double, the input is a mass and a tilt, the output
 * is a tilt rate. That is what makes the whole law testable against a synthetic
 * material without a controller_manager, and the tests in test/ are where the
 * measured behaviours of real materials are pinned. This class is the adapter:
 * it owns the subscription, the command interfaces, the joint limits, and the
 * conversion between a tilt and a joint angle.
 *
 * THE POUR JOINT AND THE GRASP. The pour joint is the one whose axis tilts the
 * vessel -- under the AG-95's SIDE grasp the last joint's roll axis runs through
 * the grasp, so vessel tilt tracks it 1:1 -- which is why `pour_joint` is a
 * parameter and not an assumption: under a top-down grasp that same rotation is
 * yaw and pours nothing.
 *
 * The grasp does NOT have to be square. Nothing here knows, or needs to know,
 * what absolute angle the vessel pours at:
 *
 *   - every angle is measured from the attitude the vessel was CARRIED in, so a
 *     beaker gripped five degrees off vertical simply reaches its onset five
 *     degrees earlier;
 *   - the onset angle is FOUND, by tilting slowly until the scale sees mass,
 *     rather than configured. A pour start angle in a config file would be a
 *     promise about a grasp nobody measured;
 *   - the park angle between trim pulses is the onset minus a margin, and it is
 *     allowed to go BELOW the carried attitude, which is what a vessel handed
 *     over already leaning needs.
 *
 * Two things the grasp still owes it. `pour_direction` says which way the joint
 * has to turn to bring the lip down, because no amount of feedback discovers
 * that safely -- guessing wrong tips the vessel away from the scale. A goal may
 * carry its own, or -- pour_reference_joints -- a configuration that shows the
 * pour, and then the vessel is tipped about whatever axis the EE turns about
 * to reach it. A replay passes its recording's deepest tilt, because recordings
 * do not agree on the axis: the first real-cell ones rolled the wrist about the
 * approach, later ones turn about the jaws' closing axis. And a grasp
 * rotated about a DIFFERENT axis than the pour axis is out of scope: the joint
 * would swing the lip sideways instead of tipping it.
 *
 * TWO GEOMETRIES, chosen by `pour_geometry`:
 *
 *   joint     the pour joint alone turns; the lip swings on an arc about the
 *             grasp. Where the stream lands then depends on where the jaws
 *             closed, and the pre-pour pose has to have allowed for it.
 *   measured  the vessel's position in the gripper is MEASURED when a goal
 *             starts -- a marker on it, located by the side cameras through
 *             cho_object_pose -- and the whole arm moves so that the vessel
 *             tips about its LIP, held at one height above a receiver fixed at
 *             a configured place. pour/lip_path.hpp decides how far in over
 *             the mouth the lip may come at each tilt; this class only follows
 *             it, by damped least-squares IK on the command, never on the
 *             measured joints.
 *
 * In `measured` a goal runs Measure -> Level -> Align -> the law -> Untilt ->
 * Unalign -> Unlevel -> Return, and the arm finishes in exactly the joint
 * configuration it started in. Level (`geometry.level`, pour/level.hpp) turns the whole EE about
 * the point the jaws hold the vessel at until the vessel stands upright -- by
 * its tag's normal when the pose node publishes one for the vessel in the
 * jaws, else by the axis the grasp gave it -- because the lip path assumes it
 * does. It is skipped when the vessel leans less than level.deadband. Measure waits for a marker pose that arrived at least vessel_settle_sec
 * after the arm last moved, because the pose node aggregates over a window and
 * a window that straddles a motion describes nowhere the vessel ever was --
 * unless the goal carries a grasp measured earlier (grasp_joints and
 * grasp_marker), which a replay takes right after the jaws close, with the
 * vessel still on the bench and close to a camera. Then nothing is waited for:
 * the marker's offset from the EE then is its offset now. The
 * law is told nothing new: its tilt is still the rotation from the carried
 * attitude, and its tilt bound is capped where the lip stops being over the
 * mouth.
 *
 * Safety, in the order it is checked every cycle:
 *   - a run of physically impossible readings stops the tilt. The pan being
 *     knocked looks like this, not like silence.
 *   - a stale scale stops the tilt. A pour with no weight feedback is not an
 *     adaptive pour, and continuing to tilt on the last number it heard is the
 *     failure that empties a vessel onto the bench.
 *   - the tilt never leaves the goal's bound around the carried attitude.
 *   - the commanded angle stays inside the joint's own limits.
 *   - no single cycle moves the command by more than max_delta_q.
 *   - a non-finite command is never written; the previous one is held.
 *
 * However the pour ends -- reached, capped, timed out, cancelled, scale lost --
 * the vessel is brought back to the attitude it was carried in before the goal
 * finishes. The steps after a pour carry the vessel to its placement, and they
 * were planned for one held the way it started.
 */
class PouringController : public cho_controller::fr5::FR5BaseController
{
public:
    [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration() const override;
    CallbackReturn on_init() override;
    CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override;
    CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override;
    CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override;
    controller_interface::return_type update(const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
    //: Idle -> Pouring (the planner drives) -> Untilt (this class drives the
    //: vessel home) -> Idle. The planner never commands the final return: it has
    //: finished deciding by then, and the return is a safety obligation rather
    //: than part of the law.
    //: `measured` adds Measure (wait for the marker, plan the lip path), Level
    //: (stand the vessel upright, when it is not) and Align (move the lip to
    //: where the tilt starts) before the law, and Unalign, Unlevel (Level
    //: backwards, as slowly) and Return (joint space, to the exact start
    //: configuration) after the untilt. Any failure after Measure goes through
    //: them too; one the arm cannot follow goes straight to Return.
    enum class Phase { Idle, Measure, Level, Align, Pouring, Untilt, Unalign, Unlevel, Return };

    //: Latest marker pose, stamped on arrival on this node's clock -- the same
    //: clock the motion that invalidates it is timed on.
    struct VesselSample {
        double stamp{0.0};
        Eigen::Vector3d position{Eigen::Vector3d::Zero()};
        bool frame_ok{false};
        bool finite{false};
    };

    //: Latest attitude of the vessel in the jaws, as its tag's normal says,
    //: stamped on arrival like VesselSample.
    struct AttitudeSample {
        double stamp{0.0};
        Eigen::Vector3d axis{Eigen::Vector3d::UnitZ()};
        bool frame_ok{false};
        bool finite{false};
    };

    bool assign_parameters();
    bool assign_geometry_parameters();
    [[nodiscard]] pour::PourRequest make_request() const;
    //: One law step: the tilt rate it asks for, or 0 once it has finished (in
    //: which case begin_untilt() has been called). Where it is headed, when it
    //: is headed somewhere, is left in law_target_tilt_.
    double step_law(double now);
    //: The tilt step the law's command comes to once the tilt's speed and
    //: acceleration are bounded: a rate is followed, a target is braked into.
    double shaped_law_step(double rate, double dt);
    //: False when the goal was refused before anything moved.
    bool start_goal(double now);
    void start_pour(double now);
    controller_interface::return_type update_measured(double now, double dt);
    //: Why `sample` cannot be planned from yet; empty when it can.
    [[nodiscard]] std::string vessel_sample_problem(const VesselSample & sample, double now) const;
    //: One DLS step of the command toward `target`. False when the arm has
    //: failed to follow for ik_fail_sec, or the solve went non-finite.
    bool track(const Eigen::Isometry3d & target, double now, std::string & why);
    //: Move the tilt to `proposed` and the lip toward the path's limit there,
    //: holding the tilt instead while the lip still has to back out.
    void advance_tilt(double proposed, double dt);
    //: How fast the tilt may go in `direction` (its sign) from where it is so
    //: that neither the IK nor the lip falls behind: the joints' per-cycle
    //: bound over the joint motion a radian of tilt takes here, and the lip's
    //: speed over how fast the path's inset limit closes in. Infinite when
    //: neither binds. Kept for the trace.
    double tilt_rate_cap(double direction, double dt);
    //: The path's inset limit at `tilt` to the nearest kLipTableStep, computed
    //: the first time a pour asks for that cell and kept: for the look-ahead,
    //: never for the step the tilt takes, which is checked exactly.
    [[nodiscard]] double lip_limit_cached(double tilt) const;
    mutable std::vector<double> lip_table_;
    double lip_table_lo_{0.0};
    //: The far end of the stretch of tilt the lip looks over from `from` in
    //: `direction`: kLipLookahead, or less where the motion ends sooner.
    [[nodiscard]] double lip_window_end(double from, double direction) const;
    //: Where the lip heads: the lowest over that stretch of the path's inset
    //: limit plus kLipEnvelopeSlope per radian still to go, so it is out
    //: before the limit closes in and gets there at an even pace.
    [[nodiscard]] double lip_ahead(double from, double direction) const;
    //: The most any joint moves per radian of tilt along the lip path at
    //: (tilt, inset), by the last IK step's own solve; 0 before there is one.
    [[nodiscard]] double joints_per_tilt(double tilt, double inset) const;
    [[nodiscard]] bool vessel_sample_usable(const VesselSample & sample, double now) const;
    [[nodiscard]] bool attitude_usable(const AttitudeSample & sample, double now) const;
    //: The horizontal across the jaws at `H`, EE frame, signed so a positive
    //: tilt brings the fingertip side down; its heading is the fingertips'.
    bool fingertips_axis(const pinocchio::SE3 & H, Eigen::Vector3d & axis_ee, double & heading,
                         std::string & why) const;
    //: Measure's last step, and Level's: the lip path from the command the arm
    //: holds now, with the vessel where held_ puts it. False with `why`.
    bool plan_pour_geometry(std::string & why);
    void publish_feedback_if_due(double dt);
    //: A failure after Measure: the arm is somewhere else by now, so it goes
    //: back the way it came before the goal reports.
    void fail_geometric(const std::string & reason, Phase via);
    void abort_before_motion(const std::string & reason);
    bool read_profile(const std::string & prefix, pour::MaterialProfile & out, std::string & why);
    pour::PourLimits read_limits(const std::string & prefix) const;
    void declare_limits(const std::string & prefix, const pour::PourLimits & defaults);
    void write_command(const Eigen::VectorXd & q_cmd);
    void hold_reference();
    //: What actually goes to the hardware: `q` with the shaker's tap on the
    //: pour joint. The IK and the tilt never see the tap -- q_ref_ is the
    //: untapped command -- so a tap cannot read as the arm falling behind.
    void send(const Eigen::VectorXd & q);
    //: The pour joint's velocity limit from the description [rad/s]; infinite
    //: when it gives none.
    [[nodiscard]] double pour_joint_velocity_limit() const;
    //: The fastest the pour joint may be commanded to move [rad/s]: the lower
    //: of max_delta_q per cycle and its velocity limit. A tap and the tilt it
    //: rides on share it.
    [[nodiscard]] double pour_speed_limit(double dt) const;
    void begin_untilt(bool succeeded, const std::string & reason);
    void finish();

    std::shared_ptr<FR5PourActionServer> action_server_;

    // ---- parameters ----
    std::string scale_topic_{"/scale/reading"};
    std::string pour_joint_;
    //: +1 or -1: which way the pour joint turns to bring the lip down.
    double pour_direction_{1.0};
    //: The one this goal pours with: the goal's, when it gives one.
    double goal_direction_{1.0};
    //: How a granular pour tips: "goal" as a liquid does -- about the goal's
    //: reference configuration when it gives one -- or "pour_joint", rolling
    //: about the pour joint's own axis whatever the goal says; and which way,
    //: 0 meaning pour_direction.
    std::string granular_tilt_about_{"goal"};
    //: How a liquid tips: "goal" (the goal's reference configuration, else the
    //: pour joint) or "fingertips" (toward the fingertips, about the horizontal
    //: across the jaws, whatever the goal's reference shows).
    std::string liquid_tilt_about_{"goal"};
    double granular_pour_direction_{0.0};
    //: The lip's height above the receiver's rim for a granular pour [m]; 0
    //: is geometry.lip_height, the liquid's.
    double granular_lip_height_{0.0};
    //: How far inside the receiver's near rim a granular pour's lip may come
    //: [m]; 0 is geometry.lip_inset, the liquid's.
    double granular_lip_inset_{0.0};
    //: The tilt bound of a granular goal that sets none [rad]; 0 is max_tilt,
    //: the liquid's.
    double granular_max_tilt_{0.0};
    //: The timeout of a granular goal that sets none [s]; 0 is the liquid's.
    double granular_timeout_{0.0};
    //: The pour axis in the EE frame, from the goal's reference configuration,
    //: when it gave one (its sign tips the lip down); otherwise unused.
    Eigen::Vector3d reference_axis_ee_{Eigen::Vector3d::UnitZ()};
    bool reference_axis_valid_{false};
    double scale_timeout_{0.5};
    double max_tilt_{2.0};
    //: Lower end of the tilt range: how far BEHIND the carried attitude the
    //: vessel may be turned. The upper end comes from the goal.
    double max_back_tilt_{0.3};
    double weight_tolerance_{0.5};
    double max_delta_q_{0.005};
    //: Bound on how fast the tilt's rate may change [rad/s^2]. Every tilt the
    //: pour makes -- the law's, and the way back to the carried attitude --
    //: goes through tilt_ramp_ with it.
    double tilt_accel_{0.4};
    //: Bound on how fast the tilt's acceleration may change [rad/s^3]; 0 keeps
    //: the trapezoid. tilt_ramp_ only.
    double tilt_jerk_{0.0};
    //: The IK residual, as a multiple of its tolerance, past which the tilt is
    //: held where it is. Between 1 and this the path is only braked -- the
    //: speed scale is already 0 there -- so a residual that crosses the
    //: tolerance for a cycle slows the tilt instead of stopping it dead. 1 is
    //: the old hold at the tolerance itself.
    double ik_hold_ratio_{1.0};
    double feedback_period_{0.1};
    //: Largest believable jump between two scale samples, as a multiple of what
    //: the fastest configured flow could deliver in one sample period.
    double outlier_factor_{3.0};
    double scale_sample_period_{0.2};
    pour::PlannerConfig planner_config_;
    pour::MaterialProfile liquid_;
    pour::MaterialProfile granular_;

    // ---- measured geometry ----
    bool measured_geometry_{false};
    std::string vessel_pose_topic_;
    std::string vessel_pose_frame_{"base_link"};
    double vessel_settle_sec_{1.0};
    double vessel_pose_max_age_{1.0};
    double vessel_pose_timeout_{5.0};
    double align_speed_{0.02};
    //: The same bound for the lip's run along the align path [m/s^2].
    double align_accel_{0.02};
    double max_lip_speed_{0.03};
    double ik_lambda_{0.02};
    double ik_tolerance_{0.002};
    double ik_rot_tolerance_{0.02};
    double ik_fail_sec_{0.5};
    //: How far a vessel measured upright at its grasp may lean by the time the
    //: pour starts [rad]. The lip path assumes it hangs upright; a carry that
    //: tipped it would put the lip somewhere the path does not know about.
    double max_carry_lean_{0.10};
    //: Standing the vessel upright before the pour (geometry.level).
    bool level_enabled_{false};
    //: The vessel-in-the-jaws attitude topic; empty: level by the kinematics.
    std::string attitude_topic_;
    //: Refuse the pour when no usable attitude arrives, instead of levelling
    //: by the kinematics.
    bool level_require_tag_{false};
    double level_rate_{0.05};
    pour::LevelConfig level_config_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr attitude_sub_;
    realtime_tools::RealtimeBuffer<AttitudeSample> attitude_buffer_;
    //: The receiver's rim height in the base frame [m], replacing the goal's
    //: z; 0 takes the goal's. x and y are always the goal's.
    double receiver_rim_z_{0.0};
    pour::HeldVessel vessel_;
    pour::Receiver receiver_;
    pour::LipPathConfig lip_config_;
    pour::LipPath lip_path_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr vessel_sub_;
    realtime_tools::RealtimeBuffer<VesselSample> vessel_buffer_;

    // ---- scale ----
    rclcpp::Subscription<sdl_cell_interfaces::msg::ScaleReading>::SharedPtr scale_sub_;
    realtime_tools::RealtimeBuffer<pour::ScaleFilter::Sample> scale_buffer_;
    pour::ScaleFilter filter_;
    double last_pushed_stamp_{0.0};

    // ---- law ----
    //: Both laws are constructed and configured, whichever runs, so a bad
    //: parameter for either is refused at configure rather than when someone
    //: first switches `law` on the robot. law_ points at the one selected.
    pour::PourPlanner planner_;
    pour::ShapingPourLaw shaping_;
    pour::PourLaw * law_{nullptr};
    std::string law_name_{"phase_machine"};
    pour::ShapingConfig shaping_config_;

    // ---- goal state ----
    Eigen::VectorXd q_ref_;
    int pour_index_{-1};
    Phase phase_{Phase::Idle};
    //: Tilt in POUR-POSITIVE coordinates: always >= 0 when pouring, whichever
    //: way the joint actually turns. Everything in pour/ works in these, and
    //: pour_direction_ is applied only where a joint angle is formed.
    double tilt_{0.0};
    pour::Ramp tilt_ramp_;
    //: Where the law's last command was taking the tilt; NaN when it asked for
    //: a rate and nothing more.
    double law_target_tilt_{std::numeric_limits<double>::quiet_NaN()};
    double theta_start_{0.0};
    double peak_tilt_{0.0};
    double elapsed_{0.0};
    double feedback_accumulator_{0.0};
    double last_grams_{0.0};
    double last_flow_{0.0};
    std::uint8_t last_phase_{0};
    //: Which stops the log has already reported, and whether the latest has
    //: been graded against its settle.
    int logged_stops_{0};
    bool stop_graded_{true};
    bool pending_success_{false};
    std::string pending_reason_;
    //: Whether the law ran in this goal: a goal that ends in Measure or Align
    //: has no report of its own, and the law's is the previous goal's.
    bool law_started_{false};
    //: The tilt bound the law was given: the goal's, capped in `measured` by
    //: where the lip leaves the mouth, and then by reach_.
    double tilt_bound_{0.0};
    //: The furthest tilt the arm has shown it can make from this pose;
    //: infinite until the IK has been blocked on the way up.
    double reach_{std::numeric_limits<double>::infinity()};
    //: The furthest a tap may tip the vessel: the bound as the goal and the lip
    //: path set it, before reach_ -- which is a joint's limit, not the rim's.
    double tap_bound_{0.0};
    //: The law's last tilt rate: a block only says where the reach is when the
    //: law was asking for more tilt.
    double last_law_rate_{0.0};
    //: Taps for a granular bed, and what the law last asked of them.
    pour::Shaker shaker_;
    double shake_amplitude_{0.0};
    double shake_period_{0.0};
    //: The pour joint's last command as sent, tap included, and the part of it
    //: that was tap. While a tap is on, the joint's step per cycle is bounded
    //: by max_delta_q with the tap in it.
    double sent_pour_{std::numeric_limits<double>::quiet_NaN()};
    double sent_tap_{0.0};
    //: This cycle's control period, for send(), which has no period of its own.
    double nominal_period_{0.008};

    // ---- measured goal state ----
    double now_{0.0};
    //: When the command last changed. A marker pose measured before this plus
    //: vessel_settle_sec may have been averaged over a motion.
    double quiet_since_{0.0};
    double measure_started_{0.0};
    Eigen::VectorXd q_start_;
    double align_s_{0.0};
    pour::Ramp align_ramp_;
    //: How far past the receiver's near rim the lip is being held. Lags the
    //: path's limit by at most max_lip_speed, and the tilt waits for it
    //: whenever the limit requires the lip to back out.
    double inset_cmd_{0.0};
    //: The held vessel, rigid in the EE frame, once Measure has placed it.
    pour::HeldInJaws held_;
    bool held_measured_{false};
    //: Whether the goal tips toward the fingertips: the axis is found again
    //: from the levelled pose.
    bool fingertips_{false};
    pour::LevelMove level_move_;
    double level_s_{0.0};
    pour::Ramp level_ramp_;
    //: What Level did, for the pour geometry log line.
    std::string level_note_;
    bool ik_lagging_{false};
    //: The last IK step's residual as a fraction of its tolerance, the larger
    //: of position and rotation. 1 is where ik_lagging_ trips.
    double ik_lag_ratio_{0.0};
    //: Why advance_tilt last took less than it was handed: 0 it took it all,
    //: 1 held for the IK, 2 held for the lip, 3 a share of it for the lip,
    //: 4 braking for the IK (it took all of a braked step).
    int hold_reason_{0};
    //: The last IK step's task Jacobian and damped normal matrix, and the
    //: frame its error was taken in: what tilt_rate_cap() asks the path's
    //: derivative of.
    Eigen::MatrixXd ik_J_task_;
    Eigen::LDLT<Eigen::Matrix<double, 6, 6>> ik_JJt_ldlt_;
    Eigen::Matrix3d ik_R_{Eigen::Matrix3d::Identity()};
    bool ik_solve_cached_{false};
    //: The last caps tilt_rate_cap() found [rad/s]; infinite when none bound.
    double tilt_cap_ik_{std::numeric_limits<double>::infinity()};
    double tilt_cap_lip_{std::numeric_limits<double>::infinity()};
    //: Per-cycle trace of the tilt during Pouring and Untilt on ~/tilt_debug:
    //: [time, phase, tilt, proposed, ramp v, ramp a, ik_lag_ratio, hold_reason,
    //:  inset_cmd, speed scale, IK rate cap, lip rate cap] -- a cap that does
    //: not bind is -1. For finding where a pour knocks.
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr tilt_debug_pub_;
    std::unique_ptr<realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>>
        tilt_debug_rt_;
    void publish_tilt_debug(double now, double proposed);
    //: How much of its speed the path may use given how far the IK is behind:
    //: all of it up to half the tolerance, none of it at the tolerance.
    [[nodiscard]] double ik_speed_scale() const;
    double ik_bad_since_{-1.0};
    //: Since when the residual has been within reach of the tolerance, lagging
    //: or not. ik_speed_scale() throttles a path the arm cannot follow to a
    //: halt at the tolerance's edge, where ik_lagging_ flickers and keeps
    //: resetting ik_bad_since_; this is the timer the flicker does not reset.
    double ik_blocked_since_{-1.0};
    //: Blocked for longer than `factor` x ik_fail_sec.
    [[nodiscard]] bool ik_blocked(double now, double factor = 1.0) const;
    //: "j4 at its lower limit"-style list of the joints the command has pinned
    //: at a limit, for the log; empty when none is.
    [[nodiscard]] std::string joints_at_limits() const;
    [[nodiscard]] std::string blocked_reason() const;
};

} // namespace sdl_pour_controller
