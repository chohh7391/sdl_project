# Planner comparison (cuTAMP vs PDDLStream)

cuTAMP: `/home/home/sdl_ws/src/sdl_project/_2026__IEEE_Access/revision/analysis/data/rerun/rtx5080_20261001f/cutamp`  
PDDLStream: `/home/home/sdl_ws/src/sdl_project/_2026__IEEE_Access/revision/analysis/data/rerun/rtx5080_20261001f/pddlstream`  
budgets (declared): 60, 120, 180 s


## transfer

cuTAMP repetitions [0], PDDLStream planner seeds [0]; paired on [0]. Layouts checked identical on all 30 shared seeds.

- cuTAMP solved-only planning time (n=30): mean 15.87 s, SD 4.64 s, median 14.65 s [IQR 14.17-14.90], range 14.07-33.74 s
- PDDLStream solved-only planning time (n=20): mean 1.97 s, SD 2.09 s, median 1.19 s [IQR 0.56-3.06], range 0.25-8.80 s

| budget | cuTAMP success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] | PDDLStream success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] |
|---|---|---|---|---|---|---|
| 60 s | 30/30 = 100.0% [88.6, 100.0] | 15.9 s | 14.2 / 14.7 / 14.9 | 20/30 = 66.7% [48.8, 80.8] | 21.3 s | 0.8 / 3.0 / NR |
| 120 s | 30/30 = 100.0% [88.6, 100.0] | 15.9 s | 14.2 / 14.7 / 14.9 | 20/30 = 66.7% [48.8, 80.8] | 41.3 s | 0.8 / 3.0 / NR |
| 180 s | 30/30 = 100.0% [88.6, 100.0] | 15.9 s | 14.2 / 14.7 / 14.9 | 20/30 = 66.7% [48.8, 80.8] | 61.3 s | 0.8 / 3.0 / NR |

| budget | pairing | cuTAMP | PDDLStream | cuTAMP-only | PDDLStream-only | exact McNemar p |
|---|---|---|---|---|---|---|
| 60 s | rep 0 vs seed 0 | 30/30 | 20/30 | 10 | 0 | 0.00195 |
| 120 s | rep 0 vs seed 0 | 30/30 | 20/30 | 10 | 0 | 0.00195 |
| 180 s | rep 0 vs seed 0 | 30/30 | 20/30 | 10 | 0 | 0.00195 |

## move

cuTAMP repetitions [0], PDDLStream planner seeds [0]; paired on [0]. Layouts checked identical on all 30 shared seeds.

- cuTAMP solved-only planning time (n=30): mean 8.89 s, SD 0.04 s, median 8.89 s [IQR 8.87-8.92], range 8.82-8.98 s
- PDDLStream solved-only planning time (n=30): mean 0.10 s, SD 0.10 s, median 0.08 s [IQR 0.06-0.10], range 0.05-0.60 s

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
- PDDLStream solved-only planning time (n=23): mean 0.83 s, SD 0.48 s, median 0.79 s [IQR 0.61-0.84], range 0.31-2.30 s

| budget | cuTAMP success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] | PDDLStream success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] |
|---|---|---|---|---|---|---|
| 60 s | 30/30 = 100.0% [88.6, 100.0] | 12.8 s | 11.8 / 12.0 / 12.1 | 23/30 = 76.7% [59.1, 88.2] | 14.6 s | 0.6 / 0.8 / 2.3 |
| 120 s | 30/30 = 100.0% [88.6, 100.0] | 12.8 s | 11.8 / 12.0 / 12.1 | 23/30 = 76.7% [59.1, 88.2] | 28.6 s | 0.6 / 0.8 / 2.3 |
| 180 s | 30/30 = 100.0% [88.6, 100.0] | 12.8 s | 11.8 / 12.0 / 12.1 | 23/30 = 76.7% [59.1, 88.2] | 42.6 s | 0.6 / 0.8 / 2.3 |

| budget | pairing | cuTAMP | PDDLStream | cuTAMP-only | PDDLStream-only | exact McNemar p |
|---|---|---|---|---|---|---|
| 60 s | rep 0 vs seed 0 | 30/30 | 23/30 | 7 | 0 | 0.0156 |
| 120 s | rep 0 vs seed 0 | 30/30 | 23/30 | 7 | 0 | 0.0156 |
| 180 s | rep 0 vs seed 0 | 30/30 | 23/30 | 7 | 0 | 0.0156 |
