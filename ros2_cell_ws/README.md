# ros2_cell_ws — the FR5 pour cell

The real cell's robot side: the scale-closed pouring controller, the Hansung scale driver, the pour-stream
camera, and the task trees that perceive the vessels, recover an occluded one, and replay a planned trajectory
with its pour done by weight. It is an overlay workspace on the core,
[cho_robot_project](https://github.com/chohh7391/cho_robot_project), at the tag in `cell.repos`. The core's own
documents (`docs/usage_guidelines.md`, `cho_task_manager/README.md`, `templates/overlay_project/`) describe how a
project builds on it.

`COLCON_IGNORE` keeps this directory out of `~/sdl_ws`'s `colcon build`; the cell is built on its own, as below.

## Packages

| Package | Holds |
| --- | --- |
| `sdl_cell_interfaces` | `action/Pour`, `msg/PourStream`, `msg/ScaleReading` |
| `sdl_pour_controller` | `sdl_pour_controller/PouringController` (on the core's `FR5BaseController`), the pour law (`pour/`), the `Pour` action server `/controller_action_server/<controller>`; tests `test_pour`, `test_lip_path`, the `pour_ab` comparison |
| `sdl_cell_bringup` | `bringup_{mujoco,real}_cell.launch.py`, `config/{mujoco,real}/pouring_controller.yaml`, the cell's robot registry |
| `sdl_cell_tasks` | the task trees, their `config/` (perception tables, sweep tables, layouts, one example `task_params` file per task), `meshes/`, `scripts/solve_sweep_raster.py`, the tools `fr5_pour_client` and `scale_relay` |
| `sdl_pour_stream` | `pour_stream_node`: is material falling between the lip and the receiver ([its config](src/sdl_pour_stream/config/pour_stream.yaml)) |
| `hansung_scale/` | `hansung_scale_msgs`, `hansung_scale_driver`: the HS-AA RS-232 driver ([README](src/hansung_scale/README.md)) |

## Build

```bash
# The core, at the tag in cell.repos (once, and again when the pin moves)
cd ~/ros2_ws && vcs import --recursive src < ~/sdl_ws/src/sdl_project/ros2_cell_ws/cell.repos
# build it as the core's docs/installation.md says, then:
source ~/ros2_ws/install/setup.bash
cd ~/sdl_ws/src/sdl_project/ros2_cell_ws
PYTHONNOUSERSITE=1 MAKEFLAGS="-j2 -l2" colcon build --parallel-workers 2 --symlink-install --base-paths src \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
PYTHONNOUSERSITE=1 colcon test --base-paths src && colcon test-result --verbose
```

`PYTHONNOUSERSITE=1` is needed whenever `~/.local` holds numpy 2.x or a newer pytest plugin: they break the
pinocchio import and CMake's pytest check. `hansung_scale_driver` needs `python3-serial` (rosdep installs it).

## The cell's registry

The cell loads `pouring_controller`, which claims the arm's command interfaces like the core's FR5 controllers but
has no role in the core's registry. At build time `sdl_cell_bringup` copies the installed core registry, adds
`controllers.additional_arm: [pouring_controller]` to fr5 (`config/robot_config_additions.yaml`), and validates the
result with `cho_robot_config`. Sourcing this workspace sets `CHO_ROBOT_CONFIG_DIR` to that copy, so every exclusive
controller switch, the core's `guarded_mission()` abort included, takes the pour controller down. A tree that
switches to the pour controller refuses to be built when the registry it is given does not list it.

## Run

Every terminal sources this workspace (`install/setup.bash`, which brings the core's) and uses its own
`ROS_DOMAIN_ID`.

```bash
# MuJoCo: the core's FR5 bringup with the AG-95 and the lab cell scene, plus pouring_controller (inactive)
ros2 launch sdl_cell_bringup bringup_mujoco_cell.launch.py
# Real FR5 (robot IP and gripper in cho_bringup_fr5/config/real/fr5.config.yaml); moveit:=true adds MoveIt
ros2 launch sdl_cell_bringup bringup_real_cell.launch.py moveit:=true
# The scale, and its relay to the controller's ScaleReading
ros2 launch hansung_scale_driver scale.launch.py
ros2 run sdl_cell_tasks scale_relay
# One pour from where the arm is, with the controller already active
ros2 run sdl_cell_tasks fr5_pour_client --target 20 --container auto
```

MuJoCo has no scale: the controller is switchable there, and a pour needs something publishing
`sdl_cell_interfaces/ScaleReading` on `/scale/reading`.

## Tasks

FR5 only, through the core's task manager:

```bash
ros2 launch cho_task_manager run_task_manager.launch.py robot_type:=fr5 task:=<task> \
  task_params:=$(ros2 pkg prefix --share sdl_cell_tasks)/config/<task>.yaml [use_sim_time:=true]
```

| Task | Does | Its `task_params` |
| --- | --- | --- |
| `sdl_trajectory_replay` | replays a recording on `joint_trajectory_controller`; the span labelled `pouring` is poured by weight by the pour controller, which returns the arm to that span's start | `pour_controller`, `replay_pour_{grams,container,material,flow_index,timeout,marker_topic,required}` |
| `sdl_perceived_replay` | the replay, with the cameras checking the layout the recording assumes, and the `replay_watch` vessels watched while it runs | `replay_watch`, `replay_drift_tolerance` |
| `sdl_occlusion_recovery` | finds the vessels, sweeping the wrist camera over any the standing cameras cannot see | `sweep_config`, `sweep_mode`, `visibility_topic` |
| `sdl_occlusion_replay` | the recovery, then the replay | as `sdl_occlusion_recovery` |
| `sdl_vessel_detect` | latches the beaker's and the flask's detected poses; moves nothing | none |

The recording and how it is replayed are the node's own arguments, as for the core's `trajectory_replay`:
`replay_trajectory:=` (empty: the core's FR5 cell recording `traj_level_seed2`), `replay_meta:=`, `replay_layout:=`
(required when the recording assumes a layout), `home_via:=`, `replay_speed_scale:=`. Each example file in
`sdl_cell_tasks/config/` lists its task's parameters with their meaning; a key a task does not take is refused.

## Not in this workspace

The laptop's real-cell additions to its cho_robot_project checkout, which never reached GitHub: the
`beaker_transfer` and `grasp_trial` tasks, `pour_beakers.yaml`, `fr5_bench_beakers.yaml`, the recovery's
`vessel_table`, `d435.launch.py stream:=both`, and the real `pouring_controller` values for the 15 mm AG-95 flange
(`grasp_depth` 0.285, `hand_boxes` +15 mm). `scripts/real_cell/` uses them.
