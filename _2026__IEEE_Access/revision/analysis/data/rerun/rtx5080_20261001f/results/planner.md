# Planner comparison (cuTAMP vs PDDLStream)

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
