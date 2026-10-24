# cuTAMP planning calls: restarts and cuRobo candidates

Per trial, from its tamp_server log. `attempts` = optimizations run; `candidate` = which satisfying particle cuRobo planned in the successful attempt (1 = the best, the only one tried before candidates were added). Times are the driver's wall clock, solved trials only -- success rates and censored times are in the planner comparison.


## move, ground_truth (30 trials with a log)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 1 | 30 | 0 | 8.7 / 8.8 / 9.0 |

successful attempt planned with candidate: 1: 30
candidates cuRobo could not plan, all trials: 0

## stir, ground_truth (30 trials with a log)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 1 | 29 | 0 | 11.5 / 11.9 / 13.0 |
| 2 | 1 | 0 | 19.0 / 19.0 / 19.0 |

successful attempt planned with candidate: 1: 26, 2: 3, 4: 1
candidates cuRobo could not plan, all trials: 6

## transfer, ground_truth (30 trials with a log)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 1 | 30 | 0 | 14.0 / 14.2 / 15.6 |

successful attempt planned with candidate: 1: 23, 2: 6, 3: 1
candidates cuRobo could not plan, all trials: 8

## transfer, perception (30 trials with a log)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 0 | 0 | 2 | -- |
| 1 | 26 | 0 | 14.2 / 14.5 / 16.5 |
| 2 | 1 | 0 | 38.9 / 38.9 / 38.9 |
| 1472 | 0 | 1 | -- |

successful attempt planned with candidate: 1: 17, 2: 6, 3: 1, 4: 1, 5: 2
candidates cuRobo could not plan, all trials: 27
