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
| `sdl_cell_bringup` | `bringup_{mujoco,real}_cell.launch.py`, `config/{mujoco,real}/pouring_controller.yaml`, the cell's robot registry; for the real cell `config/real/camera_extrinsics.yaml` (its cameras' poses), `config/real/fr5_config_overrides.yaml` (the grip force) and `d435.launch.py` (a D435 with colour beside infra1) |
| `sdl_cell_tasks` | the task trees, their `config/` (perception tables, sweep tables, layouts, real-cell recordings in `replay/real/`, one example `task_params` file per task), `meshes/`, `scripts/solve_sweep_raster.py`, the tools `fr5_pour_client` and `scale_relay` |
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
# Real FR5: the core's bringup given cho_bringup_fr5/config/real/fr5.config.yaml (robot IP, gripper)
# with config/real/fr5_config_overrides.yaml merged over it (grip force 60 %); moveit:=true adds MoveIt
ros2 launch sdl_cell_bringup bringup_real_cell.launch.py moveit:=true
# The real cell's camera poses, through the core's launch file
ros2 launch cho_bringup_fr5 camera_extrinsics.launch.py \
  extrinsics_config:=$(ros2 pkg prefix --share sdl_cell_bringup)/config/real/camera_extrinsics.yaml
# A D435: the core's d435.launch.py with infra1 for the tags as it is, plus colour at 640x480x15 to look at
CFG=$(ros2 pkg prefix --share cho_bringup_fr5)/config/real
ros2 launch sdl_cell_bringup d435.launch.py camera_namespace:=side_2 camera_name:=side_2 \
  serial_no:=_332322072253 camera_config:=$CFG/d435_side.yaml
# The scale, and its relay to the controller's ScaleReading
ros2 launch hansung_scale_driver scale.launch.py
ros2 run sdl_cell_tasks scale_relay
# One pour from where the arm is, with the controller already active
ros2 run sdl_cell_tasks fr5_pour_client --target 20 --container auto
```

`d435.launch.py` leaves the core's launch as it is: its `stream:=both` (the default) includes the core's with
`stream:=infra1` and hands rs_launch a copy of `camera_config` with `enable_color` and `rgb_camera.color_profile`
added, which wins because rs_launch gives the node the config file after the launch arguments.

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
| `sdl_trajectory_replay` | replays a recording on `joint_trajectory_controller`; the span labelled `pouring` is poured by weight by the pour controller, which returns the arm to that span's start; `replay_pour_repeat` N pours N times there, each after the first on `/real_cell/trial_go` | `pour_controller`, `replay_pour_{grams,container,material,flow_index,timeout,marker_topic,required,receiver,grasp_marker,put_back,repeat}` |
| `sdl_beaker_transfer` | the real two-beaker cell: `sdl_trajectory_replay` with the recording, the layout gate and the pour target (beaker2's rim, beaker1's perceived grasp, put back where taken) read from one plan, `scripts/real_cell/plan_remote.sh`'s `<run>/deliver/traj_realcell_seed<N>` | `transfer_plan`, `replay_pour_{grams,material}` (required), the rest of `sdl_trajectory_replay`'s but the plan's |
| `sdl_grasp_trial` | the plan's grasp of beaker1 and nothing else: reach, close, log `grasp HELD`/`MISSED`, open, back along the reach | `transfer_plan`, `grasp_held_from` |
| `sdl_perceived_replay` | the replay, with the cameras checking the layout the recording assumes, and the `replay_watch` vessels watched while it runs | `replay_watch`, `replay_drift_tolerance` |
| `sdl_occlusion_recovery` | finds the vessels, sweeping the wrist camera over any the standing cameras cannot see; its first home goes as `home_via:=` says (MoveIt by default) | `sweep_config`, `sweep_mode`, `visibility_topic`, `vessel_table` (the object table to look for; the two-beaker cell's is `config/perception/pour_beakers.yaml` with `config/sweep/fr5_bench_beakers.yaml`) |
| `sdl_occlusion_replay` | the recovery, then the replay | as `sdl_occlusion_recovery` |
| `sdl_vessel_detect` | latches the beaker's and the flask's detected poses; moves nothing | none |

The recording and how it is replayed are the node's own arguments, as for the core's `trajectory_replay`:
`replay_trajectory:=` (empty: the core's FR5 cell recording `traj_level_seed2`), `replay_meta:=`, `replay_layout:=`
(required when the recording assumes a layout), `home_via:=`, `replay_speed_scale:=`. Each example file in
`sdl_cell_tasks/config/` lists its task's parameters with their meaning; a key a task does not take is refused.
The two plan tasks take their recording from `transfer_plan` and refuse `replay_trajectory:=`, `replay_meta:=` and
`replay_layout:=` beside it. Every replay home planned by MoveIt waits 1.5 s after its controller switch, for the
core's MoveIt bridge to see the planning scene ready again.

The real cell's pour controller (`config/real/pouring_controller.yaml`) pours under `pour_geometry: measured`: it lays
the lip out around the goal's `receiver_rim_center` and refuses a goal without one, and stands the held beaker
upright first (`geometry.level`). MuJoCo's pours about the pour joint and does not use the receiver, which is why
`sdl_trajectory_replay` does not require one.

## Session log

[SESSION_LOG.md](SESSION_LOG.md): the cell's state and the record of its sessions (pouring, the scale, the
cameras and their fusion, the recovery sweep), in Korean.

## What came from the laptop, and what did not

The real cell ran on the laptop's own cho_robot_project checkout, 29 commits off GitHub's main
(`599fbc9..5c7921d`, 2026-09-29..10-05, never pushed; exported as patches). What is the cell's is here:

| Laptop core change | Here |
| --- | --- |
| `beaker_transfer`, `grasp_trial` tasks, the pour subtree's receiver, perceived grasp and put-back, the repeat with the operator's go, the granular wait, the replay clock taken over the replayed moves only, the 1.5 s MoveIt gate wait, the recovery's MoveIt first home and `vessel_table` | `sdl_cell_tasks`: `sdl_beaker_transfer`, `sdl_grasp_trial`, `sdl_trajectory_replay`, `sdl_occlusion_recovery` |
| `pour_beakers.yaml`, `fr5_bench_beakers.yaml` | `sdl_cell_tasks/config/{perception,sweep}/` |
| the pour law and controller (measured geometry with Level, per-material lips, tilts, stops, jerk bound), `Pour.action receiver_rim_center` | `sdl_pour_controller`, `sdl_cell_interfaces` |
| the real `pouring_controller` values, the 15 mm flange's `grasp_depth` 0.285 and `hand_boxes` +15 mm among them | `sdl_cell_bringup/config/real/pouring_controller.yaml` |
| re-solved camera extrinsics | `sdl_cell_bringup/config/real/camera_extrinsics.yaml` |
| grip force 30 -> 60 % | `sdl_cell_bringup/config/real/fr5_config_overrides.yaml` |
| `d435.launch.py stream:=both` | `sdl_cell_bringup/launch/d435.launch.py`, around the core's |
| two untracked recordings, `traj_return_seed{3,4}` | `sdl_cell_tasks/config/replay/real/` |

Not here, being changes to the core itself; cho_robot_project v0.1.0 does not have them:

- **The 15 mm AG-95 adapter flange in the model** (`fr5.urdf.xacro gripper_flange`, `fr5_macro.xacro`, `fr5.srdf`). The
  pour controller's own geometry has it; MoveIt's and RViz's gripper sits 15 mm short of the real one.
- **cho_object_pose holds** (`hold_unseen`, `<topic>/release|restore|resume`, `<topic>/in_hand`, the table's
  `display:`). The v0.1.0 pose node ignores those keys in `pour_beakers.yaml`: a beaker no camera sees has no pose, the
  replay's release and restore find no service and carry on, and Level stands beaker1 up by its grasp's axis.
- **`joint_trajectory_controller` path tolerance 0.30 rad** (the core's 0.10). On the laptop it stopped the replayed
  pick's 50-70 % tracking lag from aborting and faulting the FR5 (4 aborts in 8 picks, 2026-09-29/30). The core's real
  bringup loads its controllers.yaml from a fixed path and takes no other parameter file, so the cell has nowhere to
  set it.
- **The fairino driver's ServoJ timing log and its skip when the robot's motion queue passes 6.**
