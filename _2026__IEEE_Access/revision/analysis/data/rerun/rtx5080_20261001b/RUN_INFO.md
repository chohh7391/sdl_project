# RUN_INFO — re-run campaign `rtx5080_20261001b`

Written by `scripts/rerun/00_preflight.sh` on 2026-09-29 09:12:00 KST.

| | |
|---|---|
| commit | `e1ce911acf1d0dfb755c44c451d10af59e404423` (branch `revision-access`) |
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
| cuTAMP repetitions | 3 x 30 layouts per task |
| PDDLStream | 5 planner seeds x 30 layouts, max 180 s |
| GPU exclusive at start | yes (pre-flight found no other compute process) |

## Behaviour switches, at their code defaults (no SDL_* variable was set)

Pour continuations, the recovery ladder and the planning hold were added AFTER
the 2026-09-12/13 campaign, so this run measures them and that campaign did not.
The other rows describe the same scene that campaign ran.

| switch | default | added |
|---|---|---|
| SDL_POUR_CONTINUATIONS | 5 | 09-14, re-seeded pour-path continuation |
| SDL_RECOVERY | 1 | 09-16, wait for a re-detection before a missed tag fails |
| SDL_RECOVERY_RETREAT | 1 | 09-16, retreat to home and look again |
| SDL_RECOVERY_SCAN | 0 | off: needs the wrist camera |
| SDL_PLAN_HOLD_SIM | 1 | 09-28, the simulator stops stepping (physics + rendering) while cuTAMP plans |
| SDL_UPRIGHT_TRANSPORT | 1 | upright bound on carry segments |
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
ok   grasp beta min [deg]   10.0
ok   grasp beta max [deg]   35.0
ok   grasp height bias      0.0
ok   transfer entities      ('beaker', 'flask', 'magnet')
ok   transfer statics       ('table', 'goal_region', 'stirrer', 'magnet')
```
