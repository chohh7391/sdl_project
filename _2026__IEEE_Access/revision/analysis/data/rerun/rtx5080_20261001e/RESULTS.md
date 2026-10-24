# 재실험 결과 — `rtx5080_20261001e`

`scripts/rerun/90_analyse.sh`가 2026-09-30 03:03:35에 생성했습니다. 손으로 고치지 말고 다시 생성하세요.
환경과 코드는 [RUN_INFO.md](RUN_INFO.md), 절차와 원고 반영 위치는 저장소 루트의 `RERUN.md`에 있습니다.

> **감사: CLEAN.** 모든 배치가 끝났고, 실행 중 GPU를 공유한 프로세스가 없었고, 논문 scene에서 돌았습니다.

# Re-run audit: `rtx5080_20261001e`

commit `580c948d2febbcbcef4d594105cab4c560387039`, declared budgets 60,120,180 s

## Batches

| batch | attempts | last status | wall time (last) | max load avg (last) | scene |
|---|---|---|---|---|---|
| transfer_ground_truth_rep0 | 1 | done | 0.8 h | 9.0 | paper |
| move_ground_truth_rep0 | 1 | done | 0.6 h | 10.0 | paper |
| stir_ground_truth_rep0 | 1 | done | 0.9 h | 10.4 | paper |
| transfer_perception_rep0 | 1 | done | 1.1 h | 12.8 | paper |
| pddlstream_transfer_ps0 | 1 | done | 0.2 h | 4.2 | paper |
| pddlstream_move_ps0 | 1 | done | 0.0 h | nan | paper |
| pddlstream_stir_ps0 | 1 | done | 0.0 h | 1.0 | paper |
| llm_xdl_generator | 1 | done | 0.0 h | 1.1 | paper |
| llm_ar_heldout_split2400 | 1 | done | 0.0 h | 1.0 | paper |
| llm_ar_unseen_control_in_distribution | 1 | done | 0.0 h | 0.9 | paper |
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

cuTAMP: `/home/home/sdl_ws/src/sdl_project/_2026__IEEE_Access/revision/analysis/data/rerun/rtx5080_20261001e/cutamp`  
PDDLStream: `/home/home/sdl_ws/src/sdl_project/_2026__IEEE_Access/revision/analysis/data/rerun/rtx5080_20261001e/pddlstream`  
budgets (declared): 60, 120, 180 s


## transfer

cuTAMP repetitions [0], PDDLStream planner seeds [0]; paired on [0]. Layouts checked identical on all 30 shared seeds.

- cuTAMP solved-only planning time (n=30): mean 15.60 s, SD 3.41 s, median 14.73 s [IQR 14.46-15.31], range 13.90-32.85 s
- PDDLStream solved-only planning time (n=20): mean 2.78 s, SD 2.91 s, median 1.50 s [IQR 0.64-3.94], range 0.43-10.52 s

| budget | cuTAMP success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] | PDDLStream success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] |
|---|---|---|---|---|---|---|
| 60 s | 30/30 = 100.0% [88.6, 100.0] | 15.6 s | 14.5 / 14.7 / 15.3 | 20/30 = 66.7% [48.8, 80.8] | 21.9 s | 0.8 / 3.8 / NR |
| 120 s | 30/30 = 100.0% [88.6, 100.0] | 15.6 s | 14.5 / 14.7 / 15.3 | 20/30 = 66.7% [48.8, 80.8] | 41.9 s | 0.8 / 3.8 / NR |
| 180 s | 30/30 = 100.0% [88.6, 100.0] | 15.6 s | 14.5 / 14.7 / 15.3 | 20/30 = 66.7% [48.8, 80.8] | 61.9 s | 0.8 / 3.8 / NR |

| budget | pairing | cuTAMP | PDDLStream | cuTAMP-only | PDDLStream-only | exact McNemar p |
|---|---|---|---|---|---|---|
| 60 s | rep 0 vs seed 0 | 30/30 | 20/30 | 10 | 0 | 0.00195 |
| 120 s | rep 0 vs seed 0 | 30/30 | 20/30 | 10 | 0 | 0.00195 |
| 180 s | rep 0 vs seed 0 | 30/30 | 20/30 | 10 | 0 | 0.00195 |

## move

cuTAMP repetitions [0], PDDLStream planner seeds [0]; paired on [0]. Layouts checked identical on all 30 shared seeds.

- cuTAMP solved-only planning time (n=30): mean 9.04 s, SD 0.11 s, median 9.03 s [IQR 8.95-9.12], range 8.81-9.28 s
- PDDLStream solved-only planning time (n=30): mean 0.11 s, SD 0.10 s, median 0.08 s [IQR 0.06-0.10], range 0.05-0.60 s

| budget | cuTAMP success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] | PDDLStream success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] |
|---|---|---|---|---|---|---|
| 60 s | 30/30 = 100.0% [88.6, 100.0] | 9.0 s | 8.9 / 9.0 / 9.1 | 30/30 = 100.0% [88.6, 100.0] | 0.1 s | 0.1 / 0.1 / 0.1 |
| 120 s | 30/30 = 100.0% [88.6, 100.0] | 9.0 s | 8.9 / 9.0 / 9.1 | 30/30 = 100.0% [88.6, 100.0] | 0.1 s | 0.1 / 0.1 / 0.1 |
| 180 s | 30/30 = 100.0% [88.6, 100.0] | 9.0 s | 8.9 / 9.0 / 9.1 | 30/30 = 100.0% [88.6, 100.0] | 0.1 s | 0.1 / 0.1 / 0.1 |

| budget | pairing | cuTAMP | PDDLStream | cuTAMP-only | PDDLStream-only | exact McNemar p |
|---|---|---|---|---|---|---|
| 60 s | rep 0 vs seed 0 | 30/30 | 30/30 | 0 | 0 | 1 |
| 120 s | rep 0 vs seed 0 | 30/30 | 30/30 | 0 | 0 | 1 |
| 180 s | rep 0 vs seed 0 | 30/30 | 30/30 | 0 | 0 | 1 |

## stir

cuTAMP repetitions [0], PDDLStream planner seeds [0]; paired on [0]. Layouts checked identical on all 30 shared seeds.

- cuTAMP solved-only planning time (n=30): mean 12.92 s, SD 3.42 s, median 12.14 s [IQR 11.95-12.27], range 11.71-29.78 s
- PDDLStream solved-only planning time (n=25): mean 0.77 s, SD 0.45 s, median 0.74 s [IQR 0.48-0.83], range 0.30-2.26 s

| budget | cuTAMP success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] | PDDLStream success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] |
|---|---|---|---|---|---|---|
| 60 s | 30/30 = 100.0% [88.6, 100.0] | 12.9 s | 11.9 / 12.1 / 12.3 | 25/30 = 83.3% [66.4, 92.7] | 10.6 s | 0.5 / 0.8 / 1.0 |
| 120 s | 30/30 = 100.0% [88.6, 100.0] | 12.9 s | 11.9 / 12.1 / 12.3 | 25/30 = 83.3% [66.4, 92.7] | 20.6 s | 0.5 / 0.8 / 1.0 |
| 180 s | 30/30 = 100.0% [88.6, 100.0] | 12.9 s | 11.9 / 12.1 / 12.3 | 25/30 = 83.3% [66.4, 92.7] | 30.6 s | 0.5 / 0.8 / 1.0 |

| budget | pairing | cuTAMP | PDDLStream | cuTAMP-only | PDDLStream-only | exact McNemar p |
|---|---|---|---|---|---|---|
| 60 s | rep 0 vs seed 0 | 30/30 | 25/30 | 5 | 0 | 0.0625 |
| 120 s | rep 0 vs seed 0 | 30/30 | 25/30 | 5 | 0 | 0.0625 |
| 180 s | rep 0 vs seed 0 | 30/30 | 25/30 | 5 | 0 | 0.0625 |

---

### cuTAMP planning calls: restarts and cuRobo candidates

Per trial, from its tamp_server log. `attempts` = optimizations run; `candidate` = which satisfying particle cuRobo planned in the successful attempt (1 = the best, the only one tried before candidates were added). Times are the driver's wall clock, solved trials only -- success rates and censored times are in the planner comparison.


### move, ground_truth (30 trials with a log)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 1 | 30 | 0 | 8.8 / 9.0 / 9.3 |

successful attempt planned with candidate: 1: 30
candidates cuRobo could not plan, all trials: 0

### stir, ground_truth (30 trials with a log)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 1 | 28 | 0 | 11.7 / 12.1 / 12.6 |
| 2 | 2 | 0 | 18.9 / 24.3 / 29.8 |

successful attempt planned with candidate: 1: 28, 2: 1, 3: 1
candidates cuRobo could not plan, all trials: 11

### transfer, ground_truth (30 trials with a log)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 1 | 29 | 0 | 13.9 / 14.7 / 18.0 |
| 2 | 1 | 0 | 32.9 / 32.9 / 32.9 |

successful attempt planned with candidate: 1: 19, 2: 3, 3: 5, 4: 1, 6: 1, 8: 1
candidates cuRobo could not plan, all trials: 36

### transfer, perception (30 trials with a log)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 0 | 0 | 2 | -- |
| 1 | 27 | 1 | 14.3 / 14.7 / 16.8 |

successful attempt planned with candidate: 1: 22, 2: 3, 4: 2
candidates cuRobo could not plan, all trials: 9

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
  placement error from goal centre [mm]: n=30 min 7.5  median 13.5  max 20.3
  time [s]                     n=30  min 8.81  median 9.03  mean 9.04  sd 0.11  max 9.28 
  time <= 180 s                 30/30  = 100.0%  [95% CI  88.6, 100.0]
```
### stir_ground_truth_rep0
```
  task success                  29/30  =  96.7%  [95% CI  83.3,  99.4]   <- headline [stir]: vessel upright on the stirrer + stir bar inside it
  planning success              30/30  = 100.0%  [95% CI  88.6, 100.0]
  placement error from goal centre [mm]: n=30 min 0.6  median 3.8  max 246.4
  time [s]                     n=30  min 11.71  median 12.14  mean 12.92  sd 3.42  max 29.78 
  time <= 180 s                 30/30  = 100.0%  [95% CI  88.6, 100.0]
```
### transfer_ground_truth_rep0
```
  task success                  28/30  =  93.3%  [95% CI  78.7,  98.2]   <- headline [transfer]: poured + placed upright in the goal region
  planning success              30/30  = 100.0%  [95% CI  88.6, 100.0]
  placement error from goal centre [mm]: n=30 min 5.1  median 14.8  max 199.7
  time [s]                     n=30  min 13.90  median 14.73  mean 15.60  sd 3.41  max 32.85 
  time <= 180 s                 30/30  = 100.0%  [95% CI  88.6, 100.0]
```
### transfer_perception_rep0
```
  task success                  26/30  =  86.7%  [95% CI  70.3,  94.7]   <- headline [transfer]: poured + placed upright in the goal region
  planning success              27/30  =  90.0%  [95% CI  74.4,  96.5]
  placement error from goal centre [mm]: n=27 min 4.3  median 12.5  max 117.2
  time [s]                     n=27  min 14.27  median 14.65  mean 14.71  sd 0.53  max 16.78 
  time <= 180 s                 27/27  = 100.0%  [95% CI  87.5, 100.0]
```

---

## 붓기 립 오차 (정답 상태 Transfer)
### transfer_ground_truth_rep0
```
pours: 30
  along  mean  -14.7  sd  23.0  median  -13.1  95% CI of the mean [-22.9, -6.4]  |mean|/sd 0.64  negative 26/30
  perp   mean   +6.3  sd   9.1  median   +4.8  95% CI of the mean [+3.0, +9.5]  |mean|/sd 0.69  negative 7/30
  |err|  median  16.3  p90  28.9  within 17 mm 17/30 = 57%

counterfactual: subtract the mean 'along' offset from every pour
  |err| median  16.3 ->  10.4   within 17 mm 17/30 -> 24/30

verdict: scatter dominates; no constant correction recovers much
```

---

# 언어 모델
## XDL Generator
Avg Inference Time: 1.3313s
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
Avg Inference Time: 0.0750s  (median 0.0696s, first query 0.4209s, steady-state mean 0.0744s over 599 queries)
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
Avg Inference Time: 0.0768s  (median 0.0841s, first query 0.4351s, steady-state mean 0.0756s over 299 queries)
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
Avg Inference Time: 0.0704s  (median 0.0637s, first query 0.4114s, steady-state mean 0.0702s over 1545 queries)
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
Avg Inference Time: 0.0797s  (median 0.0838s, first query 0.4237s, steady-state mean 0.0762s over 100 queries)
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
Avg Inference Time: 0.0780s  (median 0.0844s, first query 0.4378s, steady-state mean 0.0764s over 222 queries)
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
