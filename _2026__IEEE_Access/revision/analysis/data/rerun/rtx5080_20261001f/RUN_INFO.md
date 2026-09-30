# RUN_INFO — re-run campaign `rtx5080_20261001f`

Written by `scripts/rerun/00_preflight.sh` on 2026-09-30 11:34:05 KST.

| | |
|---|---|
| commit | `63bb4b0254c64f917b80d22b8d4def613120ed49` (branch `revision-access`) |
| host / user | home / home |
| GPU | NVIDIA GeForce RTX 5080, 580.178.04, 16303 MiB |
| CUDA (nvidia-smi) | CUDA Version: 13.0 |
| CPU | AMD Ryzen 7 9800X3D 8-Core Processor (16 threads) |
| RAM | 30 GB |
| OS | Ubuntu 22.04.5 LTS, kernel 6.8.0-138-generic |
| Isaac Sim | 6.0.0.1 |
| torch (conda sdl) | 2.7.0+cu128 CUDA 12.8 |
| cuRobo | 0.0.0 |
| declared budgets | 60,120,180 s |
| cuTAMP repetitions | 1 x 30 layouts per task |
| cuTAMP planning | restarts until the largest budget (SDL_PLAN_BUDGET_S=180, set per trial by 10_cutamp.sh); up to 8 satisfying particles tried in cuRobo per optimization, >= 0.1 rad apart |
| PDDLStream | 1 planner seeds x 30 layouts, max 180 s; stock adaptive algorithm, one solve() per trial, no restart |
| GPU exclusive at start | yes (pre-flight found no other compute process) |

## Behaviour switches, at their code defaults (no SDL_* variable was set)

Pour continuations, the recovery ladder and the planning hold were added AFTER
the 2026-09-12/13 campaign, so this run measures them and that campaign did not.
The other rows describe the same scene that campaign ran.

| switch | default | added |
|---|---|---|
| SDL_POUR_CONTINUATIONS | 5 | 09-14, re-seeded pour-path continuation |
| SDL_RECOVERY | 1 | 09-16, wait for a re-detection before a missed tag fails |
| SDL_RECOVERY_RETREAT | 0 | 09-16, retreat to home and look again; off since 09-30 (never moved; the scan replaces it) |
| SDL_RECOVERY_SCAN | 1 | 09-30, carry the wrist camera over six viewpoints until it sees the tag; latch that pose |
| SDL_WRIST_CAMERA | 1 | 09-30, D435 on the wrist (perception_manager config/wrist_camera.yaml) |
| SDL_VERIFY_BEFORE_EXECUTION | 1 | 09-30, fixed cameras look again after planning; a vessel > 0.015 m off is re-planned once |
| SDL_PLAN_HOLD_SIM | 1 | 09-28, the simulator stops stepping (physics + rendering) while cuTAMP plans |
| SDL_UPRIGHT_TRANSPORT | 1 | upright bound on carry segments |
| SDL_UPRIGHT_TILT_TOL_DEG | 5.0 | 09-29, 15 -> 5 deg: the paper's theta_max for the held vessel's tilt from vertical |
| SDL_TAG_MOUNT | raise | 09-12, tags on a raised mount |
| SDL_GLASSWARE | legacy | the paper's glassware |

## Scene check

```
ok   glassware set          legacy
ok   beaker dims            [0.05, 0.05, 0.135]
ok   flask dims             [0.07, 0.07, 0.12]
ok   beaker extra height    0.0
ok   bench offset           0.0
ok   beaker riser           0.0
ok   table pose z           -0.01
ok   grasp beta min [deg]   0.0
ok   grasp beta max [deg]   18.0
ok   grasp height bias      0.0
ok   wrist camera           True
ok   tag plate heights      {'beaker': 0.18, 'flask': 0.2}
ok   wrist camera mount     [0.0, 0.05038, 0.22]
ok   transfer entities      ('beaker', 'flask', 'magnet')
ok   transfer statics       ('table', 'goal_region', 'stirrer', 'magnet')
```

## PDDLStream measured again at a later commit (2026-09-30 17:36:10)

| | |
|---|---|
| commit | `c2f8bd047208315f51296ff22f1c400aa38030a0` |
| changed since `63bb4b0254c64f917b80d22b8d4def613120ed49` | `RERUN.md`, `_2026__IEEE_Access/revision/analysis/planner_attempts.py`, `experiments/pddlstream/examples/pybullet/fr5_paired/run_paired_trials.py`, `scripts/rerun/00_preflight.sh`, `scripts/rerun/20_pddlstream.sh`, `scripts/rerun/90_analyse.sh` |
| PDDLStream | 1 planner seeds x 30 layouts, max 180 s; adaptive algorithm, restarted with fresh samples until the limit is spent |

## Perception batches measured again at a later commit (2026-09-30 18:44:47)

The ground-truth batches keep `63bb4b0254c64f917b80d22b8d4def613120ed49`.

| | |
|---|---|
| commit | `a50f925987928653d640dff4434e089a0b3c2b07` |
| changed since `63bb4b0254c64f917b80d22b8d4def613120ed49` | `RERUN.md`, `TAMP/tamp/scripts/server/tamp_server.py`, `TAMP/tamp/test/test_perception_recovery.py`, `_2026__IEEE_Access/revision/analysis/planner_attempts.py`, `experiments/pddlstream/examples/pybullet/fr5_paired/run_paired_trials.py`, `perception/perception_manager/include/perception_manager/perception_manager.hpp`, `perception/perception_manager/launch/perception_manager.launch.py`, `perception/perception_manager/src/perception_manager.cpp`, `scripts/rerun/00_preflight.sh`, `scripts/rerun/10_cutamp.sh`, `scripts/rerun/20_pddlstream.sh`, `scripts/rerun/90_analyse.sh` |
