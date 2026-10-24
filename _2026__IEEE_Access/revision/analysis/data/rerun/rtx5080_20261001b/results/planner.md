# Planner comparison (cuTAMP vs PDDLStream)

cuTAMP: `/home/home/sdl_ws/src/sdl_project/_2026__IEEE_Access/revision/analysis/data/rerun/rtx5080_20261001b/cutamp`  
PDDLStream: `/home/home/sdl_ws/src/sdl_project/_2026__IEEE_Access/revision/analysis/data/rerun/rtx5080_20261001b/pddlstream`  
budgets (declared): 60, 120, 180 s


## transfer

cuTAMP repetitions [0, 1, 2], PDDLStream planner seeds [0, 1, 2, 3, 4]; paired on [0, 1, 2]. Layouts checked identical on all 30 shared seeds.

- cuTAMP solved-only planning time (n=88): mean 18.72 s, SD 8.69 s, median 14.22 s [IQR 14.05-14.84], range 13.96-43.73 s
- PDDLStream solved-only planning time (n=89): mean 2.80 s, SD 3.87 s, median 1.57 s [IQR 0.77-2.92], range 0.19-22.01 s

| budget | cuTAMP success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] | PDDLStream success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] |
|---|---|---|---|---|---|---|
| 60 s | 88/90 = 97.8% [92.3, 99.4] | 19.6 s | 14.1 / 14.2 / 28.1 | 89/150 = 59.3% [51.3, 66.9] | 26.1 s | 1.2 / 5.1 / NR |
| 120 s | 88/90 = 97.8% [92.3, 99.4] | 21.0 s | 14.1 / 14.2 / 28.1 | 89/150 = 59.3% [51.3, 66.9] | 50.5 s | 1.2 / 5.1 / NR |
| 180 s | 88/90 = 97.8% [92.3, 99.4] | 22.3 s | 14.1 / 14.2 / 28.1 | 89/150 = 59.3% [51.3, 66.9] | 74.9 s | 1.2 / 5.1 / NR |

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

- cuTAMP solved-only planning time (n=90): mean 8.81 s, SD 0.06 s, median 8.81 s [IQR 8.78-8.84], range 8.73-9.16 s
- PDDLStream solved-only planning time (n=150): mean 0.12 s, SD 0.10 s, median 0.08 s [IQR 0.07-0.11], range 0.05-0.64 s

| budget | cuTAMP success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] | PDDLStream success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] |
|---|---|---|---|---|---|---|
| 60 s | 90/90 = 100.0% [95.9, 100.0] | 8.8 s | 8.8 / 8.8 / 8.8 | 150/150 = 100.0% [97.5, 100.0] | 0.1 s | 0.1 / 0.1 / 0.1 |
| 120 s | 90/90 = 100.0% [95.9, 100.0] | 8.8 s | 8.8 / 8.8 / 8.8 | 150/150 = 100.0% [97.5, 100.0] | 0.1 s | 0.1 / 0.1 / 0.1 |
| 180 s | 90/90 = 100.0% [95.9, 100.0] | 8.8 s | 8.8 / 8.8 / 8.8 | 150/150 = 100.0% [97.5, 100.0] | 0.1 s | 0.1 / 0.1 / 0.1 |

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

- cuTAMP solved-only planning time (n=88): mean 12.96 s, SD 3.77 s, median 11.84 s [IQR 11.71-11.93], range 11.45-25.75 s
- PDDLStream solved-only planning time (n=117): mean 0.89 s, SD 0.50 s, median 0.77 s [IQR 0.49-1.18], range 0.30-3.29 s

| budget | cuTAMP success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] | PDDLStream success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] |
|---|---|---|---|---|---|---|
| 60 s | 88/90 = 97.8% [92.3, 99.4] | 14.0 s | 11.7 / 11.8 / 11.9 | 117/150 = 78.0% [70.7, 83.9] | 13.9 s | 0.6 / 0.9 / 1.8 |
| 120 s | 88/90 = 97.8% [92.3, 99.4] | 15.3 s | 11.7 / 11.8 / 11.9 | 117/150 = 78.0% [70.7, 83.9] | 27.1 s | 0.6 / 0.9 / 1.8 |
| 180 s | 88/90 = 97.8% [92.3, 99.4] | 16.7 s | 11.7 / 11.8 / 11.9 | 117/150 = 78.0% [70.7, 83.9] | 40.3 s | 0.6 / 0.9 / 1.8 |

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
