# 재실험 결과 — `rtx5080_20261001f`

`scripts/rerun/90_analyse.sh`가 2026-10-01 14:02:42에 생성했습니다. 손으로 고치지 말고 다시 생성하세요.
환경과 코드는 [RUN_INFO.md](RUN_INFO.md), 절차와 원고 반영 위치는 저장소 루트의 `RERUN.md`에 있습니다.
인식 상태 배치는 태그 커밋이 아니라 `a50f925987928653d640dff4434e089a0b3c2b07`에서 다시 쟀습니다. 정답 상태 배치는 태그 커밋 그대로입니다(RUN_INFO.md 끝).
PDDLStream의 인식 상태 실행은 `fb2bca3a745d7e524c3a3ddf2fe4b7f3f5967f66`에서 쟀습니다. 입력은 인식 상태 cuTAMP 시행이 계획에 쓴 위치입니다(RUN_INFO.md 끝).
PDDLStream은 태그 커밋이 아니라 `c2f8bd047208315f51296ff22f1c400aa38030a0`에서 다시 쟀습니다. 그 사이 바뀐 것은 기준선과 분석뿐입니다(RUN_INFO.md 끝).

> **감사: CLEAN.** 모든 배치가 끝났고, 실행 중 GPU를 공유한 프로세스가 없었고, 논문 scene에서 돌았습니다.

# Re-run audit: `rtx5080_20261001f`

commit `63bb4b0254c64f917b80d22b8d4def613120ed49`, declared budgets 60,120,180 s

## Batches

| batch | attempts | last status | wall time (last) | max load avg (last) | scene |
|---|---|---|---|---|---|
| transfer_ground_truth_rep0 | 1 | done | 0.8 h | 9.3 | paper |
| move_ground_truth_rep0 | 1 | done | 0.6 h | 9.2 | paper |
| stir_ground_truth_rep0 | 1 | done | 0.9 h | 8.2 | paper |
| transfer_perception_rep0 | 2 | done | 1.2 h | 14.3 | paper |
| move_perception_rep0 | 3 | done | 0.0 h | 7.1 | paper |
| stir_perception_rep0 | 2 | done | 1.3 h | 14.3 | paper |
| pddlstream_transfer_ps0 | 2 | done | 0.4 h | 3.5 | paper |
| pddlstream_move_ps0 | 2 | done | 0.0 h | nan | paper |
| pddlstream_stir_ps0 | 2 | done | 0.0 h | 2.1 | paper |
| llm_xdl_generator | 1 | done | 0.0 h | 2.0 | paper |
| llm_ar_heldout_split2400 | 1 | done | 0.0 h | 2.1 | paper |
| llm_ar_unseen_control_in_distribution | 1 | done | 0.0 h | 2.2 | paper |
| llm_ar_unseen_L1_unseen_class | 1 | done | 0.0 h | 2.2 | paper |
| llm_ar_unseen_L2_unseen_layout | 1 | done | 0.0 h | nan | paper |
| llm_ar_unseen_L3_unseen_pair | 1 | done | 0.0 h | 2.1 | paper |
| pddlstream_perception_transfer_ps0 | 1 | done | 0.5 h | 3.0 | paper |
| pddlstream_perception_move_ps0 | 1 | done | 0.0 h | 1.2 | paper |
| pddlstream_perception_stir_ps0 | 1 | done | 0.0 h | 1.2 | paper |

## Trials: completeness and GPU exclusivity

| file | planner seed | layouts | duplicates | missing | trials that shared the GPU |
|---|---|---|---|---|---|
| `cutamp/move_ground_truth_rep0.csv` | - | 30 | - | - | - |
| `cutamp/move_perception_rep0.csv` | - | 30 | - | - | - |
| `cutamp/stir_ground_truth_rep0.csv` | - | 30 | - | - | - |
| `cutamp/stir_perception_rep0.csv` | - | 30 | - | - | - |
| `cutamp/transfer_ground_truth_rep0.csv` | - | 30 | - | - | - |
| `cutamp/transfer_perception_rep0.csv` | - | 30 | - | - | - |
| `pddlstream/pddlstream_move_5streams.csv` | 0 | 30 | - | - | - |
| `pddlstream/pddlstream_stir_5streams.csv` | 0 | 30 | - | - | - |
| `pddlstream/pddlstream_transfer_5streams.csv` | 0 | 30 | - | - | - |
| `pddlstream_perception/pddlstream_move_5streams.csv` | 0 | 30 | - | - | - |
| `pddlstream_perception/pddlstream_stir_5streams.csv` | 0 | 30 | - | - | - |
| `pddlstream_perception/pddlstream_transfer_5streams.csv` | 0 | 30 | - | - | - |

**Verdict: CLEAN**

---

## Planner comparison (cuTAMP vs PDDLStream)

cuTAMP: `/home/home/sdl_ws/src/sdl_project/_2026__IEEE_Access/revision/analysis/data/rerun/rtx5080_20261001f/cutamp`  
PDDLStream: `/home/home/sdl_ws/src/sdl_project/_2026__IEEE_Access/revision/analysis/data/rerun/rtx5080_20261001f/pddlstream`  
budgets (declared): 60, 120, 180 s


## transfer

cuTAMP repetitions [0], PDDLStream planner seeds [0]; paired on [0]. Layouts checked identical on all 30 shared seeds.

- cuTAMP solved-only planning time (n=30): mean 15.87 s, SD 4.64 s, median 14.65 s [IQR 14.17-14.90], range 14.07-33.74 s
- PDDLStream solved-only planning time (n=22): mean 7.19 s, SD 15.35 s, median 0.97 s [IQR 0.60-3.34], range 0.25-59.50 s

| budget | cuTAMP success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] | PDDLStream success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] |
|---|---|---|---|---|---|---|
| 60 s | 30/30 = 100.0% [88.6, 100.0] | 15.9 s | 14.2 / 14.7 / 14.9 | 22/30 = 73.3% [55.6, 85.8] | 21.3 s | 0.7 / 1.5 / NR |
| 120 s | 30/30 = 100.0% [88.6, 100.0] | 15.9 s | 14.2 / 14.7 / 14.9 | 22/30 = 73.3% [55.6, 85.8] | 37.3 s | 0.7 / 1.5 / NR |
| 180 s | 30/30 = 100.0% [88.6, 100.0] | 15.9 s | 14.2 / 14.7 / 14.9 | 22/30 = 73.3% [55.6, 85.8] | 53.3 s | 0.7 / 1.5 / NR |

| budget | pairing | cuTAMP | PDDLStream | cuTAMP-only | PDDLStream-only | exact McNemar p |
|---|---|---|---|---|---|---|
| 60 s | rep 0 vs seed 0 | 30/30 | 22/30 | 8 | 0 | 0.00781 |
| 120 s | rep 0 vs seed 0 | 30/30 | 22/30 | 8 | 0 | 0.00781 |
| 180 s | rep 0 vs seed 0 | 30/30 | 22/30 | 8 | 0 | 0.00781 |

## move

cuTAMP repetitions [0], PDDLStream planner seeds [0]; paired on [0]. Layouts checked identical on all 30 shared seeds.

- cuTAMP solved-only planning time (n=30): mean 8.89 s, SD 0.04 s, median 8.89 s [IQR 8.87-8.92], range 8.82-8.98 s
- PDDLStream solved-only planning time (n=30): mean 0.11 s, SD 0.10 s, median 0.08 s [IQR 0.06-0.10], range 0.05-0.61 s

| budget | cuTAMP success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] | PDDLStream success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] |
|---|---|---|---|---|---|---|
| 60 s | 30/30 = 100.0% [88.6, 100.0] | 8.9 s | 8.9 / 8.9 / 8.9 | 30/30 = 100.0% [88.6, 100.0] | 0.1 s | 0.1 / 0.1 / 0.1 |
| 120 s | 30/30 = 100.0% [88.6, 100.0] | 8.9 s | 8.9 / 8.9 / 8.9 | 30/30 = 100.0% [88.6, 100.0] | 0.1 s | 0.1 / 0.1 / 0.1 |
| 180 s | 30/30 = 100.0% [88.6, 100.0] | 8.9 s | 8.9 / 8.9 / 8.9 | 30/30 = 100.0% [88.6, 100.0] | 0.1 s | 0.1 / 0.1 / 0.1 |

| budget | pairing | cuTAMP | PDDLStream | cuTAMP-only | PDDLStream-only | exact McNemar p |
|---|---|---|---|---|---|---|
| 60 s | rep 0 vs seed 0 | 30/30 | 30/30 | 0 | 0 | 1 |
| 120 s | rep 0 vs seed 0 | 30/30 | 30/30 | 0 | 0 | 1 |
| 180 s | rep 0 vs seed 0 | 30/30 | 30/30 | 0 | 0 | 1 |

## stir

cuTAMP repetitions [0], PDDLStream planner seeds [0]; paired on [0]. Layouts checked identical on all 30 shared seeds.

- cuTAMP solved-only planning time (n=30): mean 12.75 s, SD 3.22 s, median 12.00 s [IQR 11.84-12.09], range 11.61-28.29 s
- PDDLStream solved-only planning time (n=30): mean 1.79 s, SD 2.12 s, median 0.83 s [IQR 0.51-2.22], range 0.32-8.86 s

| budget | cuTAMP success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] | PDDLStream success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] |
|---|---|---|---|---|---|---|
| 60 s | 30/30 = 100.0% [88.6, 100.0] | 12.8 s | 11.8 / 12.0 / 12.1 | 30/30 = 100.0% [88.6, 100.0] | 1.8 s | 0.5 / 0.8 / 2.3 |
| 120 s | 30/30 = 100.0% [88.6, 100.0] | 12.8 s | 11.8 / 12.0 / 12.1 | 30/30 = 100.0% [88.6, 100.0] | 1.8 s | 0.5 / 0.8 / 2.3 |
| 180 s | 30/30 = 100.0% [88.6, 100.0] | 12.8 s | 11.8 / 12.0 / 12.1 | 30/30 = 100.0% [88.6, 100.0] | 1.8 s | 0.5 / 0.8 / 2.3 |

| budget | pairing | cuTAMP | PDDLStream | cuTAMP-only | PDDLStream-only | exact McNemar p |
|---|---|---|---|---|---|---|
| 60 s | rep 0 vs seed 0 | 30/30 | 30/30 | 0 | 0 | 1 |
| 120 s | rep 0 vs seed 0 | 30/30 | 30/30 | 0 | 0 | 1 |
| 180 s | rep 0 vs seed 0 | 30/30 | 30/30 | 0 | 0 | 1 |

---

## Planner comparison (cuTAMP vs PDDLStream), perception state

cuTAMP: `/home/home/sdl_ws/src/sdl_project/_2026__IEEE_Access/revision/analysis/data/rerun/rtx5080_20261001f/cutamp`  
PDDLStream: `/home/home/sdl_ws/src/sdl_project/_2026__IEEE_Access/revision/analysis/data/rerun/rtx5080_20261001f/pddlstream_perception`  
budgets (declared): 60, 120, 180 s


## transfer

cuTAMP repetitions [0], PDDLStream planner seeds [0]; paired on [0]. Layouts checked identical on all 30 shared seeds.

- cuTAMP solved-only planning time (n=30): mean 15.88 s, SD 3.36 s, median 15.09 s [IQR 14.83-15.59], range 14.51-33.16 s
- PDDLStream solved-only planning time (n=21): mean 5.12 s, SD 12.31 s, median 1.37 s [IQR 0.66-3.00], range 0.35-57.20 s

| budget | cuTAMP success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] | PDDLStream success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] |
|---|---|---|---|---|---|---|
| 60 s | 30/30 = 100.0% [88.6, 100.0] | 15.9 s | 14.8 / 15.1 / 15.6 | 21/30 = 70.0% [52.1, 83.3] | 21.6 s | 1.0 / 2.7 / NR |
| 120 s | 30/30 = 100.0% [88.6, 100.0] | 15.9 s | 14.8 / 15.1 / 15.6 | 21/30 = 70.0% [52.1, 83.3] | 39.6 s | 1.0 / 2.7 / NR |
| 180 s | 30/30 = 100.0% [88.6, 100.0] | 15.9 s | 14.8 / 15.1 / 15.6 | 21/30 = 70.0% [52.1, 83.3] | 57.6 s | 1.0 / 2.7 / NR |

| budget | pairing | cuTAMP | PDDLStream | cuTAMP-only | PDDLStream-only | exact McNemar p |
|---|---|---|---|---|---|---|
| 60 s | rep 0 vs seed 0 | 30/30 | 21/30 | 9 | 0 | 0.00391 |
| 120 s | rep 0 vs seed 0 | 30/30 | 21/30 | 9 | 0 | 0.00391 |
| 180 s | rep 0 vs seed 0 | 30/30 | 21/30 | 9 | 0 | 0.00391 |

## move

cuTAMP repetitions [0], PDDLStream planner seeds [0]; paired on [0]. Layouts checked identical on all 30 shared seeds.

- cuTAMP solved-only planning time (n=30): mean 9.37 s, SD 0.20 s, median 9.34 s [IQR 9.30-9.37], range 9.24-10.41 s
- PDDLStream solved-only planning time (n=30): mean 0.11 s, SD 0.10 s, median 0.08 s [IQR 0.06-0.10], range 0.05-0.60 s

| budget | cuTAMP success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] | PDDLStream success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] |
|---|---|---|---|---|---|---|
| 60 s | 30/30 = 100.0% [88.6, 100.0] | 9.4 s | 9.3 / 9.3 / 9.4 | 30/30 = 100.0% [88.6, 100.0] | 0.1 s | 0.1 / 0.1 / 0.1 |
| 120 s | 30/30 = 100.0% [88.6, 100.0] | 9.4 s | 9.3 / 9.3 / 9.4 | 30/30 = 100.0% [88.6, 100.0] | 0.1 s | 0.1 / 0.1 / 0.1 |
| 180 s | 30/30 = 100.0% [88.6, 100.0] | 9.4 s | 9.3 / 9.3 / 9.4 | 30/30 = 100.0% [88.6, 100.0] | 0.1 s | 0.1 / 0.1 / 0.1 |

| budget | pairing | cuTAMP | PDDLStream | cuTAMP-only | PDDLStream-only | exact McNemar p |
|---|---|---|---|---|---|---|
| 60 s | rep 0 vs seed 0 | 30/30 | 30/30 | 0 | 0 | 1 |
| 120 s | rep 0 vs seed 0 | 30/30 | 30/30 | 0 | 0 | 1 |
| 180 s | rep 0 vs seed 0 | 30/30 | 30/30 | 0 | 0 | 1 |

## stir

cuTAMP repetitions [0], PDDLStream planner seeds [0]; paired on [0]. Layouts checked identical on all 30 shared seeds.

- cuTAMP solved-only planning time (n=30): mean 12.98 s, SD 2.01 s, median 12.46 s [IQR 12.28-12.58], range 12.12-20.83 s
- PDDLStream solved-only planning time (n=30): mean 1.44 s, SD 1.70 s, median 0.82 s [IQR 0.50-1.80], range 0.31-8.75 s

| budget | cuTAMP success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] | PDDLStream success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] |
|---|---|---|---|---|---|---|
| 60 s | 30/30 = 100.0% [88.6, 100.0] | 13.0 s | 12.3 / 12.5 / 12.6 | 30/30 = 100.0% [88.6, 100.0] | 1.4 s | 0.5 / 0.8 / 1.9 |
| 120 s | 30/30 = 100.0% [88.6, 100.0] | 13.0 s | 12.3 / 12.5 / 12.6 | 30/30 = 100.0% [88.6, 100.0] | 1.4 s | 0.5 / 0.8 / 1.9 |
| 180 s | 30/30 = 100.0% [88.6, 100.0] | 13.0 s | 12.3 / 12.5 / 12.6 | 30/30 = 100.0% [88.6, 100.0] | 1.4 s | 0.5 / 0.8 / 1.9 |

| budget | pairing | cuTAMP | PDDLStream | cuTAMP-only | PDDLStream-only | exact McNemar p |
|---|---|---|---|---|---|---|
| 60 s | rep 0 vs seed 0 | 30/30 | 30/30 | 0 | 0 | 1 |
| 120 s | rep 0 vs seed 0 | 30/30 | 30/30 | 0 | 0 | 1 |
| 180 s | rep 0 vs seed 0 | 30/30 | 30/30 | 0 | 0 | 1 |

---

## PDDLStream: ground-truth state vs perception state

Same layouts, same planner seeds; the perception state gives the planner the beaker and flask poses (x, y, yaw) cuTAMP's perception trial planned from. Planning only.


## transfer

| budget | planner seed | ground truth [Wilson 95%] | RMST | perception [Wilson 95%] | RMST | ground truth only | perception only | exact McNemar p |
|---|---|---|---|---|---|---|---|---|
| 60 s | 0 | 22/30 = 73.3% [55.6, 85.8] | 21.3 s | 21/30 = 70.0% [52.1, 83.3] | 21.6 s | 3 [12, 13, 19] | 2 [2, 24] | 1 |
| 120 s | 0 | 22/30 = 73.3% [55.6, 85.8] | 37.3 s | 21/30 = 70.0% [52.1, 83.3] | 39.6 s | 3 [12, 13, 19] | 2 [2, 24] | 1 |
| 180 s | 0 | 22/30 = 73.3% [55.6, 85.8] | 53.3 s | 21/30 = 70.0% [52.1, 83.3] | 57.6 s | 3 [12, 13, 19] | 2 [2, 24] | 1 |

ground truth: solved-only planning time median 0.97 s (n=22); not planned: no_plan_within_budget 8

perception: solved-only planning time median 1.37 s (n=21); not planned: no_plan_within_budget 9

Perceived input (from the perception rows):

| vessel | from the fixed cameras | from the wrist recovery | shift from the layout [mm] median / max |
|---|---|---|---|
| beaker | 29 | 1 | 3.0 / 6.8 |
| flask | 30 | 0 | 2.4 / 5.5 |

## move

| budget | planner seed | ground truth [Wilson 95%] | RMST | perception [Wilson 95%] | RMST | ground truth only | perception only | exact McNemar p |
|---|---|---|---|---|---|---|---|---|
| 60 s | 0 | 30/30 = 100.0% [88.6, 100.0] | 0.1 s | 30/30 = 100.0% [88.6, 100.0] | 0.1 s | 0 | 0 | 1 |
| 120 s | 0 | 30/30 = 100.0% [88.6, 100.0] | 0.1 s | 30/30 = 100.0% [88.6, 100.0] | 0.1 s | 0 | 0 | 1 |
| 180 s | 0 | 30/30 = 100.0% [88.6, 100.0] | 0.1 s | 30/30 = 100.0% [88.6, 100.0] | 0.1 s | 0 | 0 | 1 |

ground truth: solved-only planning time median 0.08 s (n=30)

perception: solved-only planning time median 0.08 s (n=30)

Perceived input (from the perception rows):

| vessel | from the fixed cameras | from the wrist recovery | shift from the layout [mm] median / max |
|---|---|---|---|
| beaker | 29 | 1 | 2.7 / 6.9 |
| flask | 30 | 0 | 2.4 / 5.2 |

## stir

| budget | planner seed | ground truth [Wilson 95%] | RMST | perception [Wilson 95%] | RMST | ground truth only | perception only | exact McNemar p |
|---|---|---|---|---|---|---|---|---|
| 60 s | 0 | 30/30 = 100.0% [88.6, 100.0] | 1.8 s | 30/30 = 100.0% [88.6, 100.0] | 1.4 s | 0 | 0 | 1 |
| 120 s | 0 | 30/30 = 100.0% [88.6, 100.0] | 1.8 s | 30/30 = 100.0% [88.6, 100.0] | 1.4 s | 0 | 0 | 1 |
| 180 s | 0 | 30/30 = 100.0% [88.6, 100.0] | 1.8 s | 30/30 = 100.0% [88.6, 100.0] | 1.4 s | 0 | 0 | 1 |

ground truth: solved-only planning time median 0.83 s (n=30)

perception: solved-only planning time median 0.82 s (n=30)

Perceived input (from the perception rows):

| vessel | from the fixed cameras | from the wrist recovery | shift from the layout [mm] median / max |
|---|---|---|---|
| beaker | 29 | 1 | 2.7 / 6.9 |
| flask | 30 | 0 | 2.2 / 5.7 |

---

### cuTAMP planning calls: restarts and cuRobo candidates

Per trial, from its tamp_server log. `attempts` = optimizations run; `candidate` = which satisfying particle cuRobo planned in the successful attempt (1 = the best, the only one tried before candidates were added). Times are the driver's wall clock, solved trials only -- success rates and censored times are in the planner comparison.


### move, ground_truth (30 trials with a log)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 1 | 30 | 0 | 8.8 / 8.9 / 9.0 |

successful attempt planned with candidate: 1: 30
candidates cuRobo could not plan, all trials: 0

### move, perception (30 trials with a log)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 1 | 30 | 0 | 9.2 / 9.3 / 10.4 |

successful attempt planned with candidate: 1: 30
candidates cuRobo could not plan, all trials: 0

### stir, ground_truth (30 trials with a log)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 1 | 28 | 0 | 11.6 / 12.0 / 12.5 |
| 2 | 2 | 0 | 19.1 / 23.7 / 28.3 |

successful attempt planned with candidate: 1: 27, 2: 3
candidates cuRobo could not plan, all trials: 11

### stir, perception (30 trials with a log)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 1 | 29 | 0 | 12.1 / 12.5 / 20.8 |
| 2 | 1 | 0 | 19.7 / 19.7 / 19.7 |

successful attempt planned with candidate: 1: 27, 2: 1, 3: 1, 8: 1
candidates cuRobo could not plan, all trials: 10

### transfer, ground_truth (30 trials with a log)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 1 | 28 | 0 | 14.1 / 14.6 / 16.3 |
| 2 | 2 | 0 | 31.8 / 32.8 / 33.7 |

successful attempt planned with candidate: 1: 19, 2: 6, 3: 1, 4: 4
candidates cuRobo could not plan, all trials: 36

### transfer, perception (30 trials with a log)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 1 | 29 | 0 | 14.5 / 15.1 / 18.3 |
| 2 | 1 | 0 | 33.2 / 33.2 / 33.2 |

successful attempt planned with candidate: 1: 18, 2: 8, 3: 1, 4: 2, 5: 1
candidates cuRobo could not plan, all trials: 28

### PDDLStream planning calls: restarts

Per trial, from its CSV. `attempts` = solve() calls within the limit; each restart puts the scene back and draws a fresh sample stream.


### pddlstream, move, ground_truth (30 trials, restart 1)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 1 | 30 | 0 | 0.0 / 0.1 / 0.6 |

solved by the first solve() alone: 30/30; solved only after a restart: 0
the same tag's run without restarts (pddlstream_no_restart/): 30/30 planned

### pddlstream, stir, ground_truth (30 trials, restart 1)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 1 | 20 | 0 | 0.3 / 0.5 / 1.3 |
| 2 | 5 | 0 | 1.5 / 2.3 / 3.8 |
| 3-9 | 5 | 0 | 3.5 / 5.8 / 8.9 |

solved by the first solve() alone: 20/30; solved only after a restart: 10
the same tag's run without restarts (pddlstream_no_restart/): 23/30 planned

### pddlstream, transfer, ground_truth (30 trials, restart 1)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 1 | 20 | 7 | 0.3 / 1.0 / 59.5 |
| 2 | 1 | 1 | 45.0 / 45.0 / 45.0 |
| 3-9 | 1 | 0 | 14.4 / 14.4 / 14.4 |

solved by the first solve() alone: 20/30; solved only after a restart: 2
not planned, by reason: no_plan_within_budget 8
the same tag's run without restarts (pddlstream_no_restart/): 20/30 planned

### pddlstream, move, perception (30 trials, restart 1)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 1 | 30 | 0 | 0.0 / 0.1 / 0.6 |

solved by the first solve() alone: 30/30; solved only after a restart: 0

### pddlstream, stir, perception (30 trials, restart 1)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 1 | 22 | 0 | 0.3 / 0.6 / 1.9 |
| 2 | 7 | 0 | 1.6 / 2.7 / 4.3 |
| 3-9 | 1 | 0 | 8.7 / 8.7 / 8.7 |

solved by the first solve() alone: 22/30; solved only after a restart: 8

### pddlstream, transfer, perception (30 trials, restart 1)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 1 | 18 | 7 | 0.4 / 1.2 / 57.2 |
| 2 | 3 | 1 | 3.0 / 6.5 / 12.8 |
| 3-9 | 0 | 1 | -- |

solved by the first solve() alone: 18/30; solved only after a restart: 3
not planned, by reason: no_plan_within_budget 9

---

## End-to-end Transfer: 정답 상태 vs 인식 상태
### task_success: ground_truth vs perception

| condition | task_success [Wilson 95%] |
|---|---|
| ground_truth | 30/30 = 100.0% [88.6, 100.0] |
| perception | 29/30 = 96.7% [83.3, 99.4] |

| pairing | ground_truth | perception | ground_truth only | perception only | exact McNemar p |
|---|---|---|---|---|---|
| rep 0 | 30/30 | 29/30 | 1 | 0 | 1 |

---

## Perception state: localization, error, the check before execution, success


### move (30 perception trials)

| vessel | fixed cameras | wrist recovery | not localized | localization with recovery [Wilson 95%] |
|---|---|---|---|---|
| beaker | 29 | 1 | 0 | 30/30 = 100.0% [88.6, 100.0] |
| flask | 30 | 0 | 0 | 30/30 = 100.0% [88.6, 100.0] |

trials that needed the wrist scan: 1; time localizing in them [s] median / max: 18.8 / 18.8

Pose error against the simulator, median / mean / p90 / max:

| vessel | source | horizontal [mm] | height, absolute [mm] | yaw, absolute [deg] |
|---|---|---|---|---|
| beaker | fixed | 2.6 / 3.0 / 4.4 / 6.9 (n=29) | 3.4 / 3.2 / 5.1 / 8.5 (n=29) | 0.1 / 0.1 / 0.2 / 0.2 (n=29) |
| beaker | recovery | 4.2 / 4.2 / 4.2 / 4.2 (n=1) | 4.5 / 4.5 / 4.5 / 4.5 (n=1) | 0.0 / 0.0 / 0.0 / 0.0 (n=1) |
| flask | fixed | 2.4 / 2.5 / 3.8 / 5.2 (n=30) | 5.2 / 5.0 / 7.5 / 9.0 (n=30) | 0.0 / 0.1 / 0.1 / 0.3 (n=30) |

The check before execution (fixed cameras, after planning):

| vessel | verified | moved (planned again) | unseen | disagreement when seen [mm] median / max |
|---|---|---|---|---|
| beaker | 29 | 0 | 1 | 0.3 / 1.2 |
| flask | 30 | 0 | 0 | 0.3 / 0.8 |

plans redone after a vessel moved: 0

Task success, ground-truth vs perception state, same layouts:

| repetition | ground truth | perception | ground truth only | perception only | exact McNemar p |
|---|---|---|---|---|---|
| 0 | 30/30 | 30/30 | 0 | 0 | 1 |

pooled: ground truth 30/30 = 100.0% [88.6, 100.0], perception 30/30 = 100.0% [88.6, 100.0]
perception, every vessel from the fixed cameras: 29/29 = 100.0% [88.3, 100.0]
perception, a vessel recovered by the wrist scan: 1/1 = 100.0% [20.7, 100.0]

### stir (30 perception trials)

| vessel | fixed cameras | wrist recovery | not localized | localization with recovery [Wilson 95%] |
|---|---|---|---|---|
| beaker | 29 | 1 | 0 | 30/30 = 100.0% [88.6, 100.0] |
| flask | 30 | 0 | 0 | 30/30 = 100.0% [88.6, 100.0] |

trials that needed the wrist scan: 1; time localizing in them [s] median / max: 19.2 / 19.2

Pose error against the simulator, median / mean / p90 / max:

| vessel | source | horizontal [mm] | height, absolute [mm] | yaw, absolute [deg] |
|---|---|---|---|---|
| beaker | fixed | 2.6 / 3.0 / 4.2 / 6.9 (n=29) | 3.0 / 3.1 / 5.1 / 8.5 (n=29) | 0.1 / 0.1 / 0.2 / 0.2 (n=29) |
| beaker | recovery | 4.4 / 4.4 / 4.4 / 4.4 (n=1) | 4.7 / 4.7 / 4.7 / 4.7 (n=1) | 0.1 / 0.1 / 0.1 / 0.1 (n=1) |
| flask | fixed | 2.2 / 2.5 / 3.3 / 5.7 (n=30) | 5.5 / 5.1 / 7.5 / 8.0 (n=30) | 0.1 / 0.1 / 0.1 / 0.3 (n=30) |

The check before execution (fixed cameras, after planning):

| vessel | verified | moved (planned again) | unseen | disagreement when seen [mm] median / max |
|---|---|---|---|---|
| beaker | 29 | 0 | 1 | 0.3 / 1.1 |
| flask | 30 | 0 | 0 | 0.4 / 1.0 |

plans redone after a vessel moved: 0

Task success, ground-truth vs perception state, same layouts:

| repetition | ground truth | perception | ground truth only | perception only | exact McNemar p |
|---|---|---|---|---|---|
| 0 | 30/30 | 30/30 | 0 | 0 | 1 |

pooled: ground truth 30/30 = 100.0% [88.6, 100.0], perception 30/30 = 100.0% [88.6, 100.0]
perception, every vessel from the fixed cameras: 29/29 = 100.0% [88.3, 100.0]
perception, a vessel recovered by the wrist scan: 1/1 = 100.0% [20.7, 100.0]

### transfer (30 perception trials)

| vessel | fixed cameras | wrist recovery | not localized | localization with recovery [Wilson 95%] |
|---|---|---|---|---|
| beaker | 29 | 1 | 0 | 30/30 = 100.0% [88.6, 100.0] |
| flask | 30 | 0 | 0 | 30/30 = 100.0% [88.6, 100.0] |

trials that needed the wrist scan: 1; time localizing in them [s] median / max: 19.5 / 19.5

Pose error against the simulator, median / mean / p90 / max:

| vessel | source | horizontal [mm] | height, absolute [mm] | yaw, absolute [deg] |
|---|---|---|---|---|
| beaker | fixed | 3.0 / 3.1 / 4.2 / 6.9 (n=29) | 3.1 / 3.2 / 5.0 / 8.6 (n=29) | 0.1 / 0.1 / 0.2 / 0.2 (n=29) |
| beaker | recovery | 4.4 / 4.4 / 4.4 / 4.4 (n=1) | 4.7 / 4.7 / 4.7 / 4.7 (n=1) | 0.0 / 0.0 / 0.0 / 0.0 (n=1) |
| flask | fixed | 2.4 / 2.6 / 3.3 / 5.5 (n=30) | 5.6 / 5.3 / 7.5 / 8.6 (n=30) | 0.0 / 0.1 / 0.1 / 0.2 (n=30) |

The check before execution (fixed cameras, after planning):

| vessel | verified | moved (planned again) | unseen | disagreement when seen [mm] median / max |
|---|---|---|---|---|
| beaker | 29 | 0 | 1 | 0.3 / 1.1 |
| flask | 30 | 0 | 0 | 0.3 / 1.3 |

plans redone after a vessel moved: 0

Task success, ground-truth vs perception state, same layouts:

| repetition | ground truth | perception | ground truth only | perception only | exact McNemar p |
|---|---|---|---|---|---|
| 0 | 30/30 | 29/30 | 1 | 0 | 1 |

pooled: ground truth 30/30 = 100.0% [88.6, 100.0], perception 29/30 = 96.7% [83.3, 99.4]
perception, every vessel from the fixed cameras: 28/29 = 96.6% [82.8, 99.4]
perception, a vessel recovered by the wrist scan: 1/1 = 100.0% [20.7, 100.0]

---

## 시행 요약 (예산 180 s)
### move_ground_truth_rep0
```
  task success                  30/30  = 100.0%  [95% CI  88.6, 100.0]   <- headline [move]: placed upright inside the goal region (no pour step)
  planning success              30/30  = 100.0%  [95% CI  88.6, 100.0]
  placement error from goal centre [mm]: n=30 min 7.6  median 13.7  max 20.3
  time [s]                     n=30  min 8.82  median 8.89  mean 8.89  sd 0.04  max 8.98 
  time <= 180 s                 30/30  = 100.0%  [95% CI  88.6, 100.0]
```
### move_perception_rep0
```
  task success                  30/30  = 100.0%  [95% CI  88.6, 100.0]   <- headline [move]: placed upright inside the goal region (no pour step)
  planning success              30/30  = 100.0%  [95% CI  88.6, 100.0]
  placement error from goal centre [mm]: n=30 min 3.9  median 13.1  max 20.0
  time [s]                     n=30  min 9.24  median 9.34  mean 9.37  sd 0.20  max 10.41 
  time <= 180 s                 30/30  = 100.0%  [95% CI  88.6, 100.0]
```
### stir_ground_truth_rep0
```
  task success                  30/30  = 100.0%  [95% CI  88.6, 100.0]   <- headline [stir]: vessel upright on the stirrer + stir bar inside it
  planning success              30/30  = 100.0%  [95% CI  88.6, 100.0]
  placement error from goal centre [mm]: n=30 min 0.6  median 6.3  max 18.0
  time [s]                     n=30  min 11.61  median 12.00  mean 12.75  sd 3.22  max 28.29 
  time <= 180 s                 30/30  = 100.0%  [95% CI  88.6, 100.0]
```
### stir_perception_rep0
```
  task success                  30/30  = 100.0%  [95% CI  88.6, 100.0]   <- headline [stir]: vessel upright on the stirrer + stir bar inside it
  planning success              30/30  = 100.0%  [95% CI  88.6, 100.0]
  placement error from goal centre [mm]: n=30 min 0.8  median 5.4  max 24.4
  time [s]                     n=30  min 12.12  median 12.46  mean 12.98  sd 2.01  max 20.83 
  time <= 180 s                 30/30  = 100.0%  [95% CI  88.6, 100.0]
```
### transfer_ground_truth_rep0
```
  task success                  30/30  = 100.0%  [95% CI  88.6, 100.0]   <- headline [transfer]: poured + placed upright in the goal region
  planning success              30/30  = 100.0%  [95% CI  88.6, 100.0]
  placement error from goal centre [mm]: n=30 min 6.3  median 14.9  max 22.3
  time [s]                     n=30  min 14.07  median 14.65  mean 15.87  sd 4.64  max 33.74 
  time <= 180 s                 30/30  = 100.0%  [95% CI  88.6, 100.0]
```
### transfer_perception_rep0
```
  task success                  29/30  =  96.7%  [95% CI  83.3,  99.4]   <- headline [transfer]: poured + placed upright in the goal region
  planning success              30/30  = 100.0%  [95% CI  88.6, 100.0]
  placement error from goal centre [mm]: n=30 min 6.1  median 14.2  max 103.3
  time [s]                     n=30  min 14.51  median 15.09  mean 15.88  sd 3.36  max 33.16 
  time <= 180 s                 30/30  = 100.0%  [95% CI  88.6, 100.0]
```

---

## 붓기 립 오차 (정답 상태 Transfer)
### transfer_ground_truth_rep0
```
pours: 30
  along  mean   -0.6  sd  19.0  median   -7.3  95% CI of the mean [-7.4, +6.2]  |mean|/sd 0.03  negative 20/30
  perp   mean   +5.1  sd  14.3  median   +1.4  95% CI of the mean [+0.0, +10.2]  |mean|/sd 0.36  negative 14/30
  |err|  median  14.2  p90  36.5  within 17 mm 18/30 = 60%

counterfactual: subtract the mean 'along' offset from every pour
  |err| median  14.2 ->  15.4   within 17 mm 18/30 -> 17/30

verdict: scatter dominates; no constant correction recovers much
```

---

# 언어 모델
## XDL Generator
Avg Inference Time: 1.2925s
```
== all instructions
instructions scored: 100   generations parsed as <procedure>: 100.0% [96.3, 100.0]

  exact procedure match    96.0% [90.2, 98.4]   (all steps, all fields)
  step count correct       99.0% [94.6, 99.8]
  operator sequence        96.0% [90.2, 98.4]   (right operators in the right order)
  operator multiset        98.0% [93.0, 99.4]   (right operators, order ignored)

  per-step operator        97.1% [93.8, 98.7]   (n=205 steps)
  object fields           100.0% [98.4, 100.0]   (n=242)
  reagent fields          100.0% [96.3, 100.0]   (n=100)
  quantity fields         100.0% [97.7, 100.0]   (n=161)
  schema-only fields      100.0% [82.4, 100.0]   (n=18, HeatChill active)

  per-operator step accuracy (all fields of that step correct):
    Add          100.0% [96.3, 100.0]   (n=100)
    CleanVessel  100.0% [80.6, 100.0]   (n=16)
    HeatChill    100.0% [82.4, 100.0]   (n=18)
    Move         100.0% [85.1, 100.0]   (n=22)
    Stir         100.0% [85.1, 100.0]   (n=22)
    Transfer     100.0% [84.5, 100.0]   (n=21)

  ordering errors (right operators, wrong order): 2

train/test overlap: 27/100 test instructions appear verbatim in procedure_instruction_v1.jsonl

== UNSEEN in training (n=73)
instructions scored: 73   generations parsed as <procedure>: 100.0% [95.0, 100.0]

  exact procedure match    94.5% [86.7, 97.8]   (all steps, all fields)
  step count correct       98.6% [92.6, 99.8]
  operator sequence        94.5% [86.7, 97.8]   (right operators in the right order)
  operator multiset        97.3% [90.5, 99.2]   (right operators, order ignored)

  per-step operator        96.5% [92.6, 98.4]   (n=171 steps)
  object fields           100.0% [98.2, 100.0]   (n=205)
  reagent fields          100.0% [95.0, 100.0]   (n=73)
  quantity fields         100.0% [97.1, 100.0]   (n=130)
  schema-only fields      100.0% [82.4, 100.0]   (n=18, HeatChill active)

  per-operator step accuracy (all fields of that step correct):
    Add          100.0% [95.0, 100.0]   (n=73)
    CleanVessel  100.0% [80.6, 100.0]   (n=16)
    HeatChill    100.0% [82.4, 100.0]   (n=18)
    Move         100.0% [83.2, 100.0]   (n=19)
    Stir         100.0% [82.4, 100.0]   (n=18)
    Transfer     100.0% [84.5, 100.0]   (n=21)

  ordering errors (right operators, wrong order): 2

== SEEN verbatim in training (n=27)
instructions scored: 27   generations parsed as <procedure>: 100.0% [87.5, 100.0]

  exact procedure match   100.0% [87.5, 100.0]   (all steps, all fields)
  step count correct      100.0% [87.5, 100.0]
  operator sequence       100.0% [87.5, 100.0]   (right operators in the right order)
  operator multiset       100.0% [87.5, 100.0]   (right operators, order ignored)

  per-step operator       100.0% [89.8, 100.0]   (n=34 steps)
  object fields           100.0% [90.6, 100.0]   (n=37)
  reagent fields          100.0% [87.5, 100.0]   (n=27)
  quantity fields         100.0% [89.0, 100.0]   (n=31)
  schema-only fields         n/a           (n=0, HeatChill active)

  per-operator step accuracy (all fields of that step correct):
    Add          100.0% [87.5, 100.0]   (n=27)
    Move         100.0% [43.9, 100.0]   (n=3)
    Stir         100.0% [51.0, 100.0]   (n=4)

  ordering errors (right operators, wrong order): 0

```
## XDL Validator
```
benchmark: 20 valid + 160 invalid = 180 samples (seed 0)
  TP 20   FN 0      (valid samples: 20)
  FP 0   TN 160      (invalid samples: 160)
  accuracy 100.0%  Wilson 95% CI [97.9, 100.0]
  capacity_violation       20/20 = 100.0%  [83.9, 100.0]
  cross_contamination      20/20 = 100.0%  [83.9, 100.0]
  device_placement         20/20 = 100.0%  [83.9, 100.0]
  missing_attribute        20/20 = 100.0%  [83.9, 100.0]
  nonexistent_object       20/20 = 100.0%  [83.9, 100.0]
  precondition_violation   20/20 = 100.0%  [83.9, 100.0]
  undefined_attribute      20/20 = 100.0%  [83.9, 100.0]
  undefined_tag            20/20 = 100.0%  [83.9, 100.0]
```
## Action Reasoner: heldout_split2400
```
Total: 600, Format errors: 0
Avg Inference Time: 0.0755s  (median 0.0701s, first query 0.4177s, steady-state mean 0.0749s over 599 queries)
  Exact Match:      596/600 (99.3%)
  Main Tool:        600/600 (100.0%)
  Need Rearrange:   600/600 (100.0%)
  Aux Tool:         600/600 (100.0%)
  Target Grid (category match): 596/600 (99.3%)
  Target Grid (exact match):    448/600 (74.7%)
  no_rearrange: 321/321 (100.0%)
  rearrange_context: 148/152 (97.4%)
  rearrange_irrelevant: 127/127 (100.0%)
```
## Action Reasoner: unseen_control_in_distribution
```
Total: 300, Format errors: 0
Avg Inference Time: 0.0762s  (median 0.0836s, first query 0.4344s, steady-state mean 0.0750s over 299 queries)
  Exact Match:      201/300 (67.0%)
  Main Tool:        300/300 (100.0%)
  Need Rearrange:   267/300 (89.0%)
  Aux Tool:         267/300 (89.0%)
  Target Grid (category match): 201/300 (67.0%)
  Target Grid (exact match):    162/300 (54.0%)
  no_rearrange: 108/124 (87.1%)
  rearrange_context: 82/136 (60.3%)
  rearrange_irrelevant: 11/40 (27.5%)
```
## Action Reasoner: unseen_L1_unseen_class
```
Total: 1546, Format errors: 0
Avg Inference Time: 0.0692s  (median 0.0626s, first query 0.4080s, steady-state mean 0.0690s over 1545 queries)
  Exact Match:      702/1546 (45.4%)
  Main Tool:        1230/1546 (79.6%)
  Need Rearrange:   1279/1546 (82.7%)
  Aux Tool:         950/1546 (61.4%)
  Target Grid (category match): 1032/1546 (66.8%)
  Target Grid (exact match):    918/1546 (59.4%)
  no_rearrange: 696/895 (77.8%)
  rearrange_context: 0/161 (0.0%)
  rearrange_irrelevant: 6/490 (1.2%)
```
## Action Reasoner: unseen_L2_unseen_layout
```
Total: 101, Format errors: 0
Avg Inference Time: 0.0805s  (median 0.0849s, first query 0.4115s, steady-state mean 0.0772s over 100 queries)
  Exact Match:      59/101 (58.4%)
  Main Tool:        101/101 (100.0%)
  Need Rearrange:   78/101 (77.2%)
  Aux Tool:         78/101 (77.2%)
  Target Grid (category match): 59/101 (58.4%)
  Target Grid (exact match):    50/101 (49.5%)
  no_rearrange: 26/36 (72.2%)
  rearrange_context: 27/44 (61.4%)
  rearrange_irrelevant: 6/21 (28.6%)
```
## Action Reasoner: unseen_L3_unseen_pair
```
Total: 223, Format errors: 0
Avg Inference Time: 0.0774s  (median 0.0837s, first query 0.4446s, steady-state mean 0.0758s over 222 queries)
  Exact Match:      138/223 (61.9%)
  Main Tool:        223/223 (100.0%)
  Need Rearrange:   191/223 (85.7%)
  Aux Tool:         191/223 (85.7%)
  Target Grid (category match): 138/223 (61.9%)
  Target Grid (exact match):    115/223 (51.6%)
  no_rearrange: 66/79 (83.5%)
  rearrange_context: 67/111 (60.4%)
  rearrange_irrelevant: 5/33 (15.2%)
```
## Tool rule (computed, against the same labels)
```
action_reasoner_test.jsonl                           n=600   main  600/600 = 100.00%   aux  600/600 = 100.00%
control_in_distribution.jsonl                        n=300   main  300/300 = 100.00%   aux  300/300 = 100.00%
L1_unseen_class.jsonl                                n=1546  main 1546/1546 = 100.00%   aux 1546/1546 = 100.00%
L2_unseen_layout.jsonl                               n=101   main  101/101 = 100.00%   aux  101/101 = 100.00%
L3_unseen_pair.jsonl                                 n=223   main  223/223 = 100.00%   aux  223/223 = 100.00%
```
