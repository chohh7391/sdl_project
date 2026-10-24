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
