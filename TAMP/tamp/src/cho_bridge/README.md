# cho_bridge — commanding the real FR5

Planning stays in this workspace. The arm does not: on real hardware the FR5 is
owned by `~/ros2_ws/src/cho_robot_project`, which brings up the vendor hardware
interface, spawns the controllers and enforces the safety limits. This package
is the one place this workspace talks to it.

```
tamp_server (plan)  ──▶  cho_bridge  ──▶  joint_trajectory_controller  ──▶  FR5
                                     ├──▶  gripper_controller (AG-95)
                                     └──▶  pouring_controller  (not built yet)
```

| module | holds |
|---|---|
| `contract.py` | every controller name, endpoint and limit borrowed from the other repo |
| `trajectory.py` | builds `JointTrajectory` from planner output, and refuses what should not be sent |
| `executor.py` | `ChoExecutor` — the command surface (trajectory, stream, gripper, pour, switch, cancel) |
| `scale_relay.py` | RS-232 scale `WeightStamped` → plain grams |

The runnable front end is `scripts/bridge/cho_bridge_node.py`.

## Why a trajectory controller and not the old joint-state stream

In simulation `tamp_server` publishes `isaac_arm_commands` (a `JointState`) at
25 Hz and the simulated arm chases it. Doing that to the real arm would bypass
every guard the cho controllers own — the per-cycle Δq clamp, the joint-limit
clamp, the task manager's safe-abort path — because the vendor's `write()`
only rejects NaN. Going through `joint_trajectory_controller` instead costs
nothing (it is already spawned by the FR5 real bringup) and gives two
interfaces for the two different jobs:

* **`follow_joint_trajectory` action** for planned segments. It reports *why* it
  aborted (path tolerance, goal time), which a fire-and-forget stream cannot.
* **`joint_trajectory` topic** for a closed loop. Publishing replaces the
  running trajectory with no handshake, so a 25 Hz loop pays no per-step cost.

The one thing the stock controller does **not** do is clamp: it enforces
tolerances, not limits. So `validate_trajectory` runs on everything this
package sends, and it is the last gate before the arm.

## What it assumes of the cho side

All of it is named in `contract.py`. Today:

* `joint_trajectory_controller`, `joint_space_position_controller` and
  `task_space_ik_controller` exist and are spawned by
  `cho_bringup_fr5/launch/bringup_real_robot.launch.py`.
* `gripper_controller` exists only when the description is expanded with
  `gripper:=ag95`.
* **`pouring_controller` and `cho_interfaces/action/Pour` do not exist yet.**
  `ChoExecutor.pour()` is written against them and raises a clear error until
  they land. Nothing else in this package depends on them.

Recommended in `cho_bringup_fr5/config/real/controllers.yaml`:

```yaml
joint_trajectory_controller:
  ros__parameters:
    open_loop_control: true   # desired from the last command, not the measured
                              # state: a 25 Hz stream otherwise accumulates
                              # servo droop into every replacement
```

## Bringing it up

```bash
# 1. the arm (in the cho workspace)
ros2 launch cho_bringup_fr5 bringup_real_robot.launch.py \
    controller_name:=joint_trajectory_controller

# 2. the supervising tree, which owns the controller state around the session
ros2 launch cho_task_manager run_task_manager.launch.py \
    task:=fjt_handover robot_type:=fr5

# 3. the scale, if this session pours
ros2 launch hansung_scale_driver scale.launch.py
scripts/bridge/cho_bridge_node.py relay

# 4. prove the command path with one small motion before any plan runs
scripts/bridge/cho_bridge_node.py status
scripts/bridge/cho_bridge_node.py nudge j1 0.05 --duration 4
```

The task tree watches `/tamp_current_op` to know the session is alive and, on
`idle`, finished. Publish it — `ChoExecutor.announce()` / `announce_idle()` —
and publish `idle` from a `finally`: a session that dies silently is only
distinguishable from a slow one by a timeout.

## Not done here

`tamp_server.py` is **unchanged**: it still streams joint states at the
simulator. Routing it through this package is the next step, and it needs three
fixes in that file first, none of which matter until a trajectory controller
reads the message:

1. `process_plan()` assigns one `builtin_interfaces/Duration` object to every
   point, so they all end up with the last time. A trajectory controller
   rejects that (`test_cho_bridge_trajectory.py` covers both the bug and the
   check that catches it).
2. `execute_plan_cb()` returns without a response when
   `set_simulation_state` is unavailable, which on real hardware is always.
3. `joint_states.position[:6]` is a slice; an AG-95 build puts the gripper
   joint in the same message. Index by name, as `ChoExecutor.joint_positions()`
   does.

Timing also stops being decorative: `process_plan` writes `dt = 0.1` while
`execute_plan_cb` replays at `SDL_EXEC_DT = 0.04`. On this path
`time_from_start` *is* the timing, so that has to become one deliberate number
(`build_trajectory(..., time_scale=)` is where to put the choice).
