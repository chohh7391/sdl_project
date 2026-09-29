# 재실험 결과 — `rtx5080_20261001c`

`scripts/rerun/90_analyse.sh`가 2026-09-29 20:19:01에 생성했습니다. 손으로 고치지 말고 다시 생성하세요.
환경과 코드는 [RUN_INFO.md](RUN_INFO.md), 절차와 원고 반영 위치는 저장소 루트의 `RERUN.md`에 있습니다.

> **감사: CLEAN.** 모든 배치가 끝났고, 실행 중 GPU를 공유한 프로세스가 없었고, 논문 scene에서 돌았습니다.

# Re-run audit: `rtx5080_20261001c`

commit `fa3922f02bce49e46812f6775d3f86e337a57e65`, declared budgets 60,120,180 s

## Batches

| batch | attempts | last status | wall time (last) | max load avg (last) | scene |
|---|---|---|---|---|---|
| transfer_ground_truth_rep0 | 1 | done | 0.8 h | 7.9 | paper |
| move_ground_truth_rep0 | 1 | done | 0.6 h | 8.6 | paper |
| stir_ground_truth_rep0 | 1 | done | 0.8 h | 7.7 | paper |
| transfer_perception_rep0 | 1 | done | 1.2 h | 14.6 | paper |
| pddlstream_transfer_ps0 | 1 | done | 0.3 h | 6.6 | paper |
| pddlstream_move_ps0 | 1 | done | 0.0 h | nan | paper |
| pddlstream_stir_ps0 | 1 | done | 0.0 h | 1.1 | paper |
| llm_xdl_generator | 1 | done | 0.0 h | 1.1 | paper |
| llm_ar_heldout_split2400 | 1 | done | 0.0 h | 1.1 | paper |
| llm_ar_unseen_control_in_distribution | 1 | done | 0.0 h | 1.0 | paper |
| llm_ar_unseen_L1_unseen_class | 1 | done | 0.0 h | 1.1 | paper |
| llm_ar_unseen_L2_unseen_layout | 1 | done | 0.0 h | 1.1 | paper |
| llm_ar_unseen_L3_unseen_pair | 1 | done | 0.0 h | 1.1 | paper |

## Trials: completeness and GPU exclusivity

| file | planner seed | layouts | duplicates | missing | trials that shared the GPU |
|---|---|---|---|---|---|
| `move_ground_truth_rep0.csv` | - | 30 | - | - | - |
| `stir_ground_truth_rep0.csv` | - | 30 | - | - | - |
| `transfer_ground_truth_rep0.csv` | - | 30 | - | - | - |
| `transfer_perception_rep0.csv` | - | 30 | - | - | - |
| `pddlstream_move_5streams.csv` | 0 | 30 | - | - | - |
| `pddlstream_stir_5streams.csv` | 0 | 30 | - | - | - |
| `pddlstream_transfer_5streams.csv` | 0 | 30 | - | - | - |

**Verdict: CLEAN**

---

## Planner comparison (cuTAMP vs PDDLStream)

cuTAMP: `/home/home/sdl_ws/src/sdl_project/_2026__IEEE_Access/revision/analysis/data/rerun/rtx5080_20261001c/cutamp`  
PDDLStream: `/home/home/sdl_ws/src/sdl_project/_2026__IEEE_Access/revision/analysis/data/rerun/rtx5080_20261001c/pddlstream`  
budgets (declared): 60, 120, 180 s


## transfer

cuTAMP repetitions [0], PDDLStream planner seeds [0]; paired on [0]. Layouts checked identical on all 30 shared seeds.

- cuTAMP solved-only planning time (n=30): mean 14.40 s, SD 0.41 s, median 14.24 s [IQR 14.05-14.63], range 14.00-15.56 s
- PDDLStream solved-only planning time (n=18): mean 4.74 s, SD 7.08 s, median 2.13 s [IQR 1.32-3.67], range 0.25-27.86 s

| budget | cuTAMP success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] | PDDLStream success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] |
|---|---|---|---|---|---|---|
| 60 s | 30/30 = 100.0% [88.6, 100.0] | 14.4 s | 14.1 / 14.2 / 14.6 | 18/30 = 60.0% [42.3, 75.4] | 26.8 s | 2.0 / 4.1 / NR |
| 120 s | 30/30 = 100.0% [88.6, 100.0] | 14.4 s | 14.1 / 14.2 / 14.6 | 18/30 = 60.0% [42.3, 75.4] | 50.8 s | 2.0 / 4.1 / NR |
| 180 s | 30/30 = 100.0% [88.6, 100.0] | 14.4 s | 14.1 / 14.2 / 14.6 | 18/30 = 60.0% [42.3, 75.4] | 74.8 s | 2.0 / 4.1 / NR |

| budget | pairing | cuTAMP | PDDLStream | cuTAMP-only | PDDLStream-only | exact McNemar p |
|---|---|---|---|---|---|---|
| 60 s | rep 0 vs seed 0 | 30/30 | 18/30 | 12 | 0 | 0.000488 |
| 120 s | rep 0 vs seed 0 | 30/30 | 18/30 | 12 | 0 | 0.000488 |
| 180 s | rep 0 vs seed 0 | 30/30 | 18/30 | 12 | 0 | 0.000488 |

## move

cuTAMP repetitions [0], PDDLStream planner seeds [0]; paired on [0]. Layouts checked identical on all 30 shared seeds.

- cuTAMP solved-only planning time (n=30): mean 8.81 s, SD 0.05 s, median 8.80 s [IQR 8.78-8.84], range 8.73-8.99 s
- PDDLStream solved-only planning time (n=30): mean 0.11 s, SD 0.10 s, median 0.08 s [IQR 0.06-0.10], range 0.05-0.60 s

| budget | cuTAMP success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] | PDDLStream success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] |
|---|---|---|---|---|---|---|
| 60 s | 30/30 = 100.0% [88.6, 100.0] | 8.8 s | 8.8 / 8.8 / 8.8 | 30/30 = 100.0% [88.6, 100.0] | 0.1 s | 0.1 / 0.1 / 0.1 |
| 120 s | 30/30 = 100.0% [88.6, 100.0] | 8.8 s | 8.8 / 8.8 / 8.8 | 30/30 = 100.0% [88.6, 100.0] | 0.1 s | 0.1 / 0.1 / 0.1 |
| 180 s | 30/30 = 100.0% [88.6, 100.0] | 8.8 s | 8.8 / 8.8 / 8.8 | 30/30 = 100.0% [88.6, 100.0] | 0.1 s | 0.1 / 0.1 / 0.1 |

| budget | pairing | cuTAMP | PDDLStream | cuTAMP-only | PDDLStream-only | exact McNemar p |
|---|---|---|---|---|---|---|
| 60 s | rep 0 vs seed 0 | 30/30 | 30/30 | 0 | 0 | 1 |
| 120 s | rep 0 vs seed 0 | 30/30 | 30/30 | 0 | 0 | 1 |
| 180 s | rep 0 vs seed 0 | 30/30 | 30/30 | 0 | 0 | 1 |

## stir

cuTAMP repetitions [0], PDDLStream planner seeds [0]; paired on [0]. Layouts checked identical on all 30 shared seeds.

- cuTAMP solved-only planning time (n=30): mean 12.11 s, SD 1.33 s, median 11.86 s [IQR 11.73-11.92], range 11.54-18.98 s
- PDDLStream solved-only planning time (n=23): mean 0.88 s, SD 0.55 s, median 0.66 s [IQR 0.49-1.13], range 0.33-2.29 s

| budget | cuTAMP success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] | PDDLStream success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] |
|---|---|---|---|---|---|---|
| 60 s | 30/30 = 100.0% [88.6, 100.0] | 12.1 s | 11.7 / 11.9 / 11.9 | 23/30 = 76.7% [59.1, 88.2] | 14.7 s | 0.5 / 0.8 / 2.3 |
| 120 s | 30/30 = 100.0% [88.6, 100.0] | 12.1 s | 11.7 / 11.9 / 11.9 | 23/30 = 76.7% [59.1, 88.2] | 28.7 s | 0.5 / 0.8 / 2.3 |
| 180 s | 30/30 = 100.0% [88.6, 100.0] | 12.1 s | 11.7 / 11.9 / 11.9 | 23/30 = 76.7% [59.1, 88.2] | 42.7 s | 0.5 / 0.8 / 2.3 |

| budget | pairing | cuTAMP | PDDLStream | cuTAMP-only | PDDLStream-only | exact McNemar p |
|---|---|---|---|---|---|---|
| 60 s | rep 0 vs seed 0 | 30/30 | 23/30 | 7 | 0 | 0.0156 |
| 120 s | rep 0 vs seed 0 | 30/30 | 23/30 | 7 | 0 | 0.0156 |
| 180 s | rep 0 vs seed 0 | 30/30 | 23/30 | 7 | 0 | 0.0156 |

---

### cuTAMP planning calls: restarts and cuRobo candidates

Per trial, from its tamp_server log. `attempts` = optimizations run; `candidate` = which satisfying particle cuRobo planned in the successful attempt (1 = the best, the only one tried before candidates were added). Times are the driver's wall clock, solved trials only -- success rates and censored times are in the planner comparison.


### move, ground_truth (30 trials with a log)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 1 | 30 | 0 | 8.7 / 8.8 / 9.0 |

successful attempt planned with candidate: 1: 30
candidates cuRobo could not plan, all trials: 0

### stir, ground_truth (30 trials with a log)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 1 | 29 | 0 | 11.5 / 11.9 / 13.0 |
| 2 | 1 | 0 | 19.0 / 19.0 / 19.0 |

successful attempt planned with candidate: 1: 26, 2: 3, 4: 1
candidates cuRobo could not plan, all trials: 6

### transfer, ground_truth (30 trials with a log)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 1 | 30 | 0 | 14.0 / 14.2 / 15.6 |

successful attempt planned with candidate: 1: 23, 2: 6, 3: 1
candidates cuRobo could not plan, all trials: 8

### transfer, perception (30 trials with a log)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 0 | 0 | 2 | -- |
| 1 | 26 | 0 | 14.2 / 14.5 / 16.5 |
| 2 | 1 | 0 | 38.9 / 38.9 / 38.9 |
| 1472 | 0 | 1 | -- |

successful attempt planned with candidate: 1: 17, 2: 6, 3: 1, 4: 1, 5: 2
candidates cuRobo could not plan, all trials: 27

---

## End-to-end Transfer: 정답 상태 vs 인식 상태
### task_success: ground_truth vs perception

| condition | task_success [Wilson 95%] |
|---|---|
| ground_truth | 28/30 = 93.3% [78.7, 98.2] |
| perception | 26/30 = 86.7% [70.3, 94.7] |

| pairing | ground_truth | perception | ground_truth only | perception only | exact McNemar p |
|---|---|---|---|---|---|
| rep 0 | 28/30 | 26/30 | 4 | 2 | 0.688 |

---

## 시행 요약 (예산 180 s)
### move_ground_truth_rep0
```
  task success                  30/30  = 100.0%  [95% CI  88.6, 100.0]   <- headline [move]: placed upright inside the goal region (no pour step)
  planning success              30/30  = 100.0%  [95% CI  88.6, 100.0]
  placement error from goal centre [mm]: n=30 min 4.4  median 13.8  max 26.2
  time [s]                     n=30  min 8.73  median 8.80  mean 8.81  sd 0.05  max 8.99 
  time <= 180 s                 30/30  = 100.0%  [95% CI  88.6, 100.0]
```
### stir_ground_truth_rep0
```
  task success                  30/30  = 100.0%  [95% CI  88.6, 100.0]   <- headline [stir]: vessel upright on the stirrer + stir bar inside it
  planning success              30/30  = 100.0%  [95% CI  88.6, 100.0]
  placement error from goal centre [mm]: n=30 min 0.8  median 5.5  max 16.9
  time [s]                     n=30  min 11.54  median 11.86  mean 12.11  sd 1.33  max 18.98 
  time <= 180 s                 30/30  = 100.0%  [95% CI  88.6, 100.0]
```
### transfer_ground_truth_rep0
```
  task success                  28/30  =  93.3%  [95% CI  78.7,  98.2]   <- headline [transfer]: poured + placed upright in the goal region
  planning success              30/30  = 100.0%  [95% CI  88.6, 100.0]
  placement error from goal centre [mm]: n=30 min 1.1  median 12.8  max 159.6
  time [s]                     n=30  min 14.00  median 14.24  mean 14.40  sd 0.41  max 15.56 
  time <= 180 s                 30/30  = 100.0%  [95% CI  88.6, 100.0]
```
### transfer_perception_rep0
```
  task success                  26/30  =  86.7%  [95% CI  70.3,  94.7]   <- headline [transfer]: poured + placed upright in the goal region
  planning success              27/30  =  90.0%  [95% CI  74.4,  96.5]
  placement error from goal centre [mm]: n=27 min 3.7  median 12.0  max 286.5
  time [s]                     n=27  min 14.22  median 14.55  mean 15.69  sd 4.68  max 38.91 
  time <= 180 s                 27/27  = 100.0%  [95% CI  87.5, 100.0]
```

---

## 붓기 립 오차 (정답 상태 Transfer)
### transfer_ground_truth_rep0
```
pours: 30
  along  mean   -5.8  sd  18.2  median  -11.9  95% CI of the mean [-12.3, +0.7]  |mean|/sd 0.32  negative 25/30
  perp   mean   +8.3  sd  15.5  median   +2.7  95% CI of the mean [+2.7, +13.8]  |mean|/sd 0.53  negative 12/30
  |err|  median  16.9  p90  48.0  within 17 mm 15/30 = 50%

counterfactual: subtract the mean 'along' offset from every pour
  |err| median  16.9 ->  15.3   within 17 mm 15/30 -> 19/30

verdict: scatter dominates; no constant correction recovers much
```

---

# 언어 모델
## XDL Generator
Avg Inference Time: 1.3302s
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
Avg Inference Time: 0.0751s  (median 0.0699s, first query 0.4281s, steady-state mean 0.0745s over 599 queries)
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
Avg Inference Time: 0.0773s  (median 0.0847s, first query 0.4426s, steady-state mean 0.0761s over 299 queries)
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
Avg Inference Time: 0.0699s  (median 0.0633s, first query 0.4264s, steady-state mean 0.0697s over 1545 queries)
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
Avg Inference Time: 0.0797s  (median 0.0838s, first query 0.4192s, steady-state mean 0.0763s over 100 queries)
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
Avg Inference Time: 0.0790s  (median 0.0854s, first query 0.4461s, steady-state mean 0.0773s over 222 queries)
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
