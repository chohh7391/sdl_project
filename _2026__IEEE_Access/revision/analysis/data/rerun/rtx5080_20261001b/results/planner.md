# Planner comparison (cuTAMP vs PDDLStream)

cuTAMP: `/home/home/sdl_ws/src/sdl_project/_2026__IEEE_Access/revision/analysis/data/rerun/rtx5080_20261001b/cutamp`  
PDDLStream: `/home/home/sdl_ws/src/sdl_project/_2026__IEEE_Access/revision/analysis/data/rerun/rtx5080_20261001b/pddlstream`  
budgets (declared): 60, 120, 180 s


## transfer

cuTAMP repetitions [0, 1, 2], PDDLStream planner seeds [0, 1, 2, 3, 4]; paired on [0, 1, 2]. Layouts checked identical on all 30 shared seeds.

- cuTAMP solved-only planning time: median 14.23 s, range 13.96-43.73 s (n=88)
- PDDLStream solved-only planning time: median 1.57 s, range 0.19-22.01 s (n=89)

| budget | cuTAMP success [Wilson 95%] | RMST | PDDLStream success [Wilson 95%] | RMST |
|---|---|---|---|---|
| 60 s | 88/90 = 97.8% [92.3, 99.4] | 19.6 s | 89/150 = 59.3% [51.3, 66.9] | 26.1 s |
| 120 s | 88/90 = 97.8% [92.3, 99.4] | 21.0 s | 89/150 = 59.3% [51.3, 66.9] | 50.5 s |
| 180 s | 88/90 = 97.8% [92.3, 99.4] | 22.3 s | 89/150 = 59.3% [51.3, 66.9] | 74.9 s |

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

- cuTAMP solved-only planning time: median 8.81 s, range 8.73-9.16 s (n=90)
- PDDLStream solved-only planning time: median 0.08 s, range 0.05-0.64 s (n=150)

| budget | cuTAMP success [Wilson 95%] | RMST | PDDLStream success [Wilson 95%] | RMST |
|---|---|---|---|---|
| 60 s | 90/90 = 100.0% [95.9, 100.0] | 8.8 s | 150/150 = 100.0% [97.5, 100.0] | 0.1 s |
| 120 s | 90/90 = 100.0% [95.9, 100.0] | 8.8 s | 150/150 = 100.0% [97.5, 100.0] | 0.1 s |
| 180 s | 90/90 = 100.0% [95.9, 100.0] | 8.8 s | 150/150 = 100.0% [97.5, 100.0] | 0.1 s |

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

- cuTAMP solved-only planning time: median 11.84 s, range 11.45-25.75 s (n=88)
- PDDLStream solved-only planning time: median 0.77 s, range 0.30-3.29 s (n=117)

| budget | cuTAMP success [Wilson 95%] | RMST | PDDLStream success [Wilson 95%] | RMST |
|---|---|---|---|---|
| 60 s | 88/90 = 97.8% [92.3, 99.4] | 14.0 s | 117/150 = 78.0% [70.7, 83.9] | 13.9 s |
| 120 s | 88/90 = 97.8% [92.3, 99.4] | 15.3 s | 117/150 = 78.0% [70.7, 83.9] | 27.1 s |
| 180 s | 88/90 = 97.8% [92.3, 99.4] | 16.7 s | 117/150 = 78.0% [70.7, 83.9] | 40.3 s |

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
