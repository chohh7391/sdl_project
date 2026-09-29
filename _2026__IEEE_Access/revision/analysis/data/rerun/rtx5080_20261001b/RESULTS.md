# 재실험 결과 — `rtx5080_20261001b`

`scripts/rerun/90_analyse.sh`가 2026-09-29 11:47:33에 생성했습니다. 손으로 고치지 말고 다시 생성하세요.
환경과 코드는 [RUN_INFO.md](RUN_INFO.md), 절차와 원고 반영 위치는 저장소 루트의 `RERUN.md`에 있습니다.

> **감사: CLEAN.** 모든 배치가 끝났고, 실행 중 GPU를 공유한 프로세스가 없었고, 논문 scene에서 돌았습니다.

# Re-run audit: `rtx5080_20261001b`

commit `e1ce911acf1d0dfb755c44c451d10af59e404423`, declared budgets 60,120,180 s

## Batches

| batch | attempts | last status | wall time (last) | max load avg (last) | scene |
|---|---|---|---|---|---|
| transfer_ground_truth_rep0 | 2 | done | 0.0 h | 4.5 | paper |
| move_ground_truth_rep0 | 2 | done | 0.0 h | 7.9 | paper |
| stir_ground_truth_rep0 | 2 | done | 0.1 h | 8.0 | paper |
| transfer_ground_truth_rep1 | 1 | done | 0.8 h | 9.6 | paper |
| move_ground_truth_rep1 | 1 | done | 0.6 h | 8.4 | paper |
| stir_ground_truth_rep1 | 1 | done | 0.8 h | 9.7 | paper |
| transfer_ground_truth_rep2 | 1 | done | 0.8 h | 8.3 | paper |
| move_ground_truth_rep2 | 1 | done | 0.6 h | 9.3 | paper |
| stir_ground_truth_rep2 | 1 | done | 0.8 h | 9.5 | paper |
| transfer_perception_rep0 | 2 | done | 0.0 h | 10.5 | paper |
| transfer_perception_rep1 | 2 | done | 0.0 h | 11.9 | paper |
| transfer_perception_rep2 | 2 | done | 0.0 h | 7.4 | paper |
| pddlstream_transfer_ps0 | 2 | done | 0.4 h | 3.1 | paper |
| pddlstream_transfer_ps1 | 2 | done | 0.2 h | 1.1 | paper |
| pddlstream_transfer_ps2 | 2 | done | 0.3 h | 1.1 | paper |
| pddlstream_transfer_ps3 | 2 | done | 0.2 h | 2.9 | paper |
| pddlstream_transfer_ps4 | 2 | done | 0.3 h | 2.1 | paper |
| pddlstream_move_ps0 | 2 | done | 0.0 h | nan | paper |
| pddlstream_move_ps1 | 2 | done | 0.0 h | 1.0 | paper |
| pddlstream_move_ps2 | 2 | done | 0.0 h | nan | paper |
| pddlstream_move_ps3 | 2 | done | 0.0 h | nan | paper |
| pddlstream_move_ps4 | 2 | done | 0.0 h | nan | paper |
| pddlstream_stir_ps0 | 2 | done | 0.0 h | 1.0 | paper |
| pddlstream_stir_ps1 | 2 | done | 0.0 h | 1.1 | paper |
| pddlstream_stir_ps2 | 2 | done | 0.0 h | 1.0 | paper |
| pddlstream_stir_ps3 | 2 | done | 0.0 h | 1.1 | paper |
| pddlstream_stir_ps4 | 2 | done | 0.0 h | 1.1 | paper |
| llm_xdl_generator | 1 | done | 0.1 h | 7.5 | paper |
| llm_ar_heldout_split2400 | 1 | done | 0.0 h | 1.3 | paper |
| llm_ar_unseen_control_in_distribution | 1 | done | 0.0 h | 0.9 | paper |
| llm_ar_unseen_L1_unseen_class | 1 | done | 0.0 h | 1.0 | paper |
| llm_ar_unseen_L2_unseen_layout | 1 | done | 0.0 h | nan | paper |
| llm_ar_unseen_L3_unseen_pair | 1 | done | 0.0 h | 1.1 | paper |

## Trials: completeness and GPU exclusivity

| file | planner seed | layouts | duplicates | missing | trials that shared the GPU |
|---|---|---|---|---|---|
| `move_ground_truth_rep0.csv` | - | 30 | - | - | - |
| `move_ground_truth_rep1.csv` | - | 30 | - | - | - |
| `move_ground_truth_rep2.csv` | - | 30 | - | - | - |
| `stir_ground_truth_rep0.csv` | - | 30 | - | - | - |
| `stir_ground_truth_rep1.csv` | - | 30 | - | - | - |
| `stir_ground_truth_rep2.csv` | - | 30 | - | - | - |
| `transfer_ground_truth_rep0.csv` | - | 30 | - | - | - |
| `transfer_ground_truth_rep1.csv` | - | 30 | - | - | - |
| `transfer_ground_truth_rep2.csv` | - | 30 | - | - | - |
| `transfer_perception_rep0.csv` | - | 30 | - | - | - |
| `transfer_perception_rep1.csv` | - | 30 | - | - | - |
| `transfer_perception_rep2.csv` | - | 30 | - | - | - |
| `pddlstream_move_5streams.csv` | 0 | 30 | - | - | - |
| `pddlstream_move_5streams.csv` | 1 | 30 | - | - | - |
| `pddlstream_move_5streams.csv` | 2 | 30 | - | - | - |
| `pddlstream_move_5streams.csv` | 3 | 30 | - | - | - |
| `pddlstream_move_5streams.csv` | 4 | 30 | - | - | - |
| `pddlstream_stir_5streams.csv` | 0 | 30 | - | - | - |
| `pddlstream_stir_5streams.csv` | 1 | 30 | - | - | - |
| `pddlstream_stir_5streams.csv` | 2 | 30 | - | - | - |
| `pddlstream_stir_5streams.csv` | 3 | 30 | - | - | - |
| `pddlstream_stir_5streams.csv` | 4 | 30 | - | - | - |
| `pddlstream_transfer_5streams.csv` | 0 | 30 | - | - | - |
| `pddlstream_transfer_5streams.csv` | 1 | 30 | - | - | - |
| `pddlstream_transfer_5streams.csv` | 2 | 30 | - | - | - |
| `pddlstream_transfer_5streams.csv` | 3 | 30 | - | - | - |
| `pddlstream_transfer_5streams.csv` | 4 | 30 | - | - | - |

**Verdict: CLEAN**

---

## Planner comparison (cuTAMP vs PDDLStream)

cuTAMP: `/home/home/sdl_ws/src/sdl_project/_2026__IEEE_Access/revision/analysis/data/rerun/rtx5080_20261001b/cutamp`  
PDDLStream: `/home/home/sdl_ws/src/sdl_project/_2026__IEEE_Access/revision/analysis/data/rerun/rtx5080_20261001b/pddlstream`  
budgets (declared): 60, 120, 180 s


## transfer

cuTAMP repetitions [0, 1, 2], PDDLStream planner seeds [0, 1, 2, 3, 4]; paired on [0, 1, 2]. Layouts checked identical on all 30 shared seeds.

- cuTAMP solved-only planning time: median 14.23 s, range 13.96-43.73 s (n=88)
- PDDLStream solved-only planning time: median 1.57 s, range 0.19-22.01 s (n=89)

| budget | cuTAMP success [Wilson 95%] | RMST | PDDLStream success [Wilson 95%] | RMST |
|---|---|---|---|---|
| 60 s | 88/90 = 97.8% [92.3, 99.4] | 19.6 s | 89/150 = 59.3% [51.3, 66.9] | 26.1 s |
| 120 s | 88/90 = 97.8% [92.3, 99.4] | 21.0 s | 89/150 = 59.3% [51.3, 66.9] | 50.5 s |
| 180 s | 88/90 = 97.8% [92.3, 99.4] | 22.3 s | 89/150 = 59.3% [51.3, 66.9] | 74.9 s |

| budget | pairing | cuTAMP | PDDLStream | cuTAMP-only | PDDLStream-only | exact McNemar p |
|---|---|---|---|---|---|---|
| 60 s | rep 0 vs seed 0 | 29/30 | 18/30 | 11 | 0 | 0.000977 |
| 60 s | rep 1 vs seed 1 | 30/30 | 20/30 | 10 | 0 | 0.00195 |
| 60 s | rep 2 vs seed 2 | 29/30 | 18/30 | 12 | 1 | 0.00342 |
| 120 s | rep 0 vs seed 0 | 29/30 | 18/30 | 11 | 0 | 0.000977 |
| 120 s | rep 1 vs seed 1 | 30/30 | 20/30 | 10 | 0 | 0.00195 |
| 120 s | rep 2 vs seed 2 | 29/30 | 18/30 | 12 | 1 | 0.00342 |
| 180 s | rep 0 vs seed 0 | 29/30 | 18/30 | 11 | 0 | 0.000977 |
| 180 s | rep 1 vs seed 1 | 30/30 | 20/30 | 10 | 0 | 0.00195 |
| 180 s | rep 2 vs seed 2 | 29/30 | 18/30 | 12 | 1 | 0.00342 |

## move

cuTAMP repetitions [0, 1, 2], PDDLStream planner seeds [0, 1, 2, 3, 4]; paired on [0, 1, 2]. Layouts checked identical on all 30 shared seeds.

- cuTAMP solved-only planning time: median 8.81 s, range 8.73-9.16 s (n=90)
- PDDLStream solved-only planning time: median 0.08 s, range 0.05-0.64 s (n=150)

| budget | cuTAMP success [Wilson 95%] | RMST | PDDLStream success [Wilson 95%] | RMST |
|---|---|---|---|---|
| 60 s | 90/90 = 100.0% [95.9, 100.0] | 8.8 s | 150/150 = 100.0% [97.5, 100.0] | 0.1 s |
| 120 s | 90/90 = 100.0% [95.9, 100.0] | 8.8 s | 150/150 = 100.0% [97.5, 100.0] | 0.1 s |
| 180 s | 90/90 = 100.0% [95.9, 100.0] | 8.8 s | 150/150 = 100.0% [97.5, 100.0] | 0.1 s |

| budget | pairing | cuTAMP | PDDLStream | cuTAMP-only | PDDLStream-only | exact McNemar p |
|---|---|---|---|---|---|---|
| 60 s | rep 0 vs seed 0 | 30/30 | 30/30 | 0 | 0 | 1 |
| 60 s | rep 1 vs seed 1 | 30/30 | 30/30 | 0 | 0 | 1 |
| 60 s | rep 2 vs seed 2 | 30/30 | 30/30 | 0 | 0 | 1 |
| 120 s | rep 0 vs seed 0 | 30/30 | 30/30 | 0 | 0 | 1 |
| 120 s | rep 1 vs seed 1 | 30/30 | 30/30 | 0 | 0 | 1 |
| 120 s | rep 2 vs seed 2 | 30/30 | 30/30 | 0 | 0 | 1 |
| 180 s | rep 0 vs seed 0 | 30/30 | 30/30 | 0 | 0 | 1 |
| 180 s | rep 1 vs seed 1 | 30/30 | 30/30 | 0 | 0 | 1 |
| 180 s | rep 2 vs seed 2 | 30/30 | 30/30 | 0 | 0 | 1 |

## stir

cuTAMP repetitions [0, 1, 2], PDDLStream planner seeds [0, 1, 2, 3, 4]; paired on [0, 1, 2]. Layouts checked identical on all 30 shared seeds.

- cuTAMP solved-only planning time: median 11.84 s, range 11.45-25.75 s (n=88)
- PDDLStream solved-only planning time: median 0.77 s, range 0.30-3.29 s (n=117)

| budget | cuTAMP success [Wilson 95%] | RMST | PDDLStream success [Wilson 95%] | RMST |
|---|---|---|---|---|
| 60 s | 88/90 = 97.8% [92.3, 99.4] | 14.0 s | 117/150 = 78.0% [70.7, 83.9] | 13.9 s |
| 120 s | 88/90 = 97.8% [92.3, 99.4] | 15.3 s | 117/150 = 78.0% [70.7, 83.9] | 27.1 s |
| 180 s | 88/90 = 97.8% [92.3, 99.4] | 16.7 s | 117/150 = 78.0% [70.7, 83.9] | 40.3 s |

| budget | pairing | cuTAMP | PDDLStream | cuTAMP-only | PDDLStream-only | exact McNemar p |
|---|---|---|---|---|---|---|
| 60 s | rep 0 vs seed 0 | 30/30 | 26/30 | 4 | 0 | 0.125 |
| 60 s | rep 1 vs seed 1 | 29/30 | 26/30 | 4 | 1 | 0.375 |
| 60 s | rep 2 vs seed 2 | 29/30 | 24/30 | 5 | 0 | 0.0625 |
| 120 s | rep 0 vs seed 0 | 30/30 | 26/30 | 4 | 0 | 0.125 |
| 120 s | rep 1 vs seed 1 | 29/30 | 26/30 | 4 | 1 | 0.375 |
| 120 s | rep 2 vs seed 2 | 29/30 | 24/30 | 5 | 0 | 0.0625 |
| 180 s | rep 0 vs seed 0 | 30/30 | 26/30 | 4 | 0 | 0.125 |
| 180 s | rep 1 vs seed 1 | 29/30 | 26/30 | 4 | 1 | 0.375 |
| 180 s | rep 2 vs seed 2 | 29/30 | 24/30 | 5 | 0 | 0.0625 |

---

## End-to-end Transfer: 정답 상태 vs 인식 상태
### task_success: ground_truth vs perception

| condition | task_success [Wilson 95%] |
|---|---|
| ground_truth | 85/90 = 94.4% [87.6, 97.6] |
| perception | 80/90 = 88.9% [80.7, 93.9] |

| pairing | ground_truth | perception | ground_truth only | perception only | exact McNemar p |
|---|---|---|---|---|---|
| rep 0 | 29/30 | 26/30 | 4 | 1 | 0.375 |
| rep 1 | 29/30 | 27/30 | 3 | 1 | 0.625 |
| rep 2 | 27/30 | 27/30 | 3 | 3 | 1 |

---

## 시행 요약 (예산 180 s)
### move_ground_truth_rep0
```
  task success                  30/30  = 100.0%  [95% CI  88.6, 100.0]   <- headline [move]: placed upright inside the goal region (no pour step)
  planning success              30/30  = 100.0%  [95% CI  88.6, 100.0]
  placement error from goal centre [mm]: n=30 min 7.2  median 13.6  max 30.3
  time [s]                     n=30  min 8.74  median 8.80  mean 8.81  sd 0.04  max 8.92 
  time <= 180 s                 30/30  = 100.0%  [95% CI  88.6, 100.0]
```
### move_ground_truth_rep1
```
  task success                  30/30  = 100.0%  [95% CI  88.6, 100.0]   <- headline [move]: placed upright inside the goal region (no pour step)
  planning success              30/30  = 100.0%  [95% CI  88.6, 100.0]
  placement error from goal centre [mm]: n=30 min 8.2  median 12.8  max 30.5
  time [s]                     n=30  min 8.74  median 8.82  mean 8.81  sd 0.03  max 8.86 
  time <= 180 s                 30/30  = 100.0%  [95% CI  88.6, 100.0]
```
### move_ground_truth_rep2
```
  task success                  30/30  = 100.0%  [95% CI  88.6, 100.0]   <- headline [move]: placed upright inside the goal region (no pour step)
  planning success              30/30  = 100.0%  [95% CI  88.6, 100.0]
  placement error from goal centre [mm]: n=30 min 5.6  median 12.2  max 22.2
  time [s]                     n=30  min 8.73  median 8.82  mean 8.82  sd 0.08  max 9.16 
  time <= 180 s                 30/30  = 100.0%  [95% CI  88.6, 100.0]
```
### stir_ground_truth_rep0
```
  task success                  29/30  =  96.7%  [95% CI  83.3,  99.4]   <- headline [stir]: vessel upright on the stirrer + stir bar inside it
  planning success              30/30  = 100.0%  [95% CI  88.6, 100.0]
  placement error from goal centre [mm]: n=30 min 0.7  median 6.1  max 264.3
  time [s]                     n=30  min 11.50  median 11.84  mean 13.82  sd 4.76  max 25.55 
  time <= 180 s                 30/30  = 100.0%  [95% CI  88.6, 100.0]
```
### stir_ground_truth_rep1
```
  task success                  29/30  =  96.7%  [95% CI  83.3,  99.4]   <- headline [stir]: vessel upright on the stirrer + stir bar inside it
  planning success              29/30  =  96.7%  [95% CI  83.3,  99.4]
  placement error from goal centre [mm]: n=29 min 0.8  median 7.2  max 17.7
  time [s]                     n=29  min 11.45  median 11.84  mean 11.79  sd 0.16  max 12.08 
  time <= 180 s                 29/29  = 100.0%  [95% CI  88.3, 100.0]
```
### stir_ground_truth_rep2
```
  task success                  29/30  =  96.7%  [95% CI  83.3,  99.4]   <- headline [stir]: vessel upright on the stirrer + stir bar inside it
  planning success              29/30  =  96.7%  [95% CI  83.3,  99.4]
  placement error from goal centre [mm]: n=29 min 0.1  median 7.0  max 23.9
  time [s]                     n=29  min 11.48  median 11.84  mean 13.22  sd 4.28  max 25.75 
  time <= 180 s                 29/29  = 100.0%  [95% CI  88.3, 100.0]
```
### transfer_ground_truth_rep0
```
  task success                  29/30  =  96.7%  [95% CI  83.3,  99.4]   <- headline [transfer]: poured + placed upright in the goal region
  planning success              29/30  =  96.7%  [95% CI  83.3,  99.4]
  placement error from goal centre [mm]: n=29 min 1.1  median 11.4  max 23.5
  time [s]                     n=29  min 13.97  median 14.42  mean 19.94  sd 9.18  max 43.73 
  time <= 180 s                 29/29  = 100.0%  [95% CI  88.3, 100.0]
```
### transfer_ground_truth_rep1
```
  task success                  29/30  =  96.7%  [95% CI  83.3,  99.4]   <- headline [transfer]: poured + placed upright in the goal region
  planning success              30/30  = 100.0%  [95% CI  88.6, 100.0]
  placement error from goal centre [mm]: n=30 min 6.5  median 14.3  max 104.2
  time [s]                     n=30  min 13.96  median 14.11  mean 17.07  sd 6.99  max 43.02 
  time <= 180 s                 30/30  = 100.0%  [95% CI  88.6, 100.0]
```
### transfer_ground_truth_rep2
```
  task success                  27/30  =  90.0%  [95% CI  74.4,  96.5]   <- headline [transfer]: poured + placed upright in the goal region
  planning success              29/30  =  96.7%  [95% CI  83.3,  99.4]
  placement error from goal centre [mm]: n=29 min 5.2  median 12.6  max 188.1
  time [s]                     n=29  min 14.00  median 14.14  mean 19.21  sd 9.74  max 43.26 
  time <= 180 s                 29/29  = 100.0%  [95% CI  88.3, 100.0]
```
### transfer_perception_rep0
```
  task success                  26/30  =  86.7%  [95% CI  70.3,  94.7]   <- headline [transfer]: poured + placed upright in the goal region
  planning success              26/30  =  86.7%  [95% CI  70.3,  94.7]
  placement error from goal centre [mm]: n=26 min 3.3  median 11.3  max 23.2
  time [s]                     n=26  min 14.21  median 14.39  mean 18.91  sd 9.02  max 43.95 
  time <= 180 s                 26/26  = 100.0%  [95% CI  87.1, 100.0]
```
### transfer_perception_rep1
```
  task success                  27/30  =  90.0%  [95% CI  74.4,  96.5]   <- headline [transfer]: poured + placed upright in the goal region
  planning success              27/30  =  90.0%  [95% CI  74.4,  96.5]
  placement error from goal centre [mm]: n=27 min 8.1  median 16.0  max 23.6
  time [s]                     n=27  min 14.24  median 14.33  mean 18.19  sd 8.72  max 43.86 
  time <= 180 s                 27/27  = 100.0%  [95% CI  87.5, 100.0]
```
### transfer_perception_rep2
```
  task success                  27/30  =  90.0%  [95% CI  74.4,  96.5]   <- headline [transfer]: poured + placed upright in the goal region
  planning success              27/30  =  90.0%  [95% CI  74.4,  96.5]
  placement error from goal centre [mm]: n=27 min 3.3  median 13.2  max 23.2
  time [s]                     n=27  min 14.18  median 14.51  mean 18.18  sd 7.52  max 42.97 
  time <= 180 s                 27/27  = 100.0%  [95% CI  87.5, 100.0]
```

---

## 붓기 립 오차 (정답 상태 Transfer)
### transfer_ground_truth_rep0
```
pours: 29
  along  mean   -3.4  sd  17.4  median   -9.3  95% CI of the mean [-9.7, +3.0]  |mean|/sd 0.19  negative 22/29
  perp   mean   +8.2  sd  13.1  median   +6.9  95% CI of the mean [+3.5, +13.0]  |mean|/sd 0.63  negative 10/29
  |err|  median  14.7  p90  39.5  within 17 mm 16/29 = 55%

counterfactual: subtract the mean 'along' offset from every pour
  |err| median  14.7 ->  15.6   within 17 mm 16/29 -> 18/29

verdict: scatter dominates; no constant correction recovers much
```
### transfer_ground_truth_rep1
```
pours: 30
  along  mean  -15.6  sd  29.0  median  -11.9  95% CI of the mean [-26.0, -5.2]  |mean|/sd 0.54  negative 27/30
  perp   mean   +3.1  sd  14.6  median   +1.6  95% CI of the mean [-2.1, +8.4]  |mean|/sd 0.21  negative 12/30
  |err|  median  16.9  p90  28.4  within 17 mm 16/30 = 53%

counterfactual: subtract the mean 'along' offset from every pour
  |err| median  16.9 ->  11.2   within 17 mm 16/30 -> 22/30

verdict: scatter dominates; no constant correction recovers much
```
### transfer_ground_truth_rep2
```
pours: 29
  along  mean   -9.0  sd  27.5  median   -5.7  95% CI of the mean [-19.0, +1.0]  |mean|/sd 0.33  negative 22/29
  perp   mean  +11.0  sd  32.9  median   +4.3  95% CI of the mean [-1.0, +22.9]  |mean|/sd 0.33  negative 9/29
  |err|  median  13.2  p90  35.2  within 17 mm 21/29 = 72%

counterfactual: subtract the mean 'along' offset from every pour
  |err| median  13.2 ->  13.3   within 17 mm 21/29 -> 20/29

verdict: scatter dominates; no constant correction recovers much
```

---

# 언어 모델
## XDL Generator
Avg Inference Time: 1.3311s
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
Avg Inference Time: 0.0751s  (median 0.0698s, first query 0.4180s, steady-state mean 0.0746s over 599 queries)
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
Avg Inference Time: 0.0769s  (median 0.0843s, first query 0.4373s, steady-state mean 0.0757s over 299 queries)
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
Avg Inference Time: 0.0700s  (median 0.0633s, first query 0.4177s, steady-state mean 0.0698s over 1545 queries)
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
Avg Inference Time: 0.0796s  (median 0.0839s, first query 0.4139s, steady-state mean 0.0763s over 100 queries)
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
Avg Inference Time: 0.0780s  (median 0.0841s, first query 0.4405s, steady-state mean 0.0764s over 222 queries)
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
