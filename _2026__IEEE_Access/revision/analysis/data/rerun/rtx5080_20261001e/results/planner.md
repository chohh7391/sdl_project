# Planner comparison (cuTAMP vs PDDLStream)

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
