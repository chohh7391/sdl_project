# Planner comparison (cuTAMP vs PDDLStream)

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
