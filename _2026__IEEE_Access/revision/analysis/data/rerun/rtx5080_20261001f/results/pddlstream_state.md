# PDDLStream: ground-truth state vs perception state

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
