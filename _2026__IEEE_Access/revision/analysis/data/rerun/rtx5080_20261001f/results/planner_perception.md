# Planner comparison (cuTAMP vs PDDLStream), perception state

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
