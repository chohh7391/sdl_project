# cuTAMP planning calls: restarts and cuRobo candidates

Per trial, from its tamp_server log. `attempts` = optimizations run; `candidate` = which satisfying particle cuRobo planned in the successful attempt (1 = the best, the only one tried before candidates were added). Times are the driver's wall clock, solved trials only -- success rates and censored times are in the planner comparison.


## move, ground_truth (30 trials with a log)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 1 | 30 | 0 | 8.8 / 9.0 / 9.3 |

successful attempt planned with candidate: 1: 30
candidates cuRobo could not plan, all trials: 0

## stir, ground_truth (30 trials with a log)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 1 | 28 | 0 | 11.7 / 12.1 / 12.6 |
| 2 | 2 | 0 | 18.9 / 24.3 / 29.8 |

successful attempt planned with candidate: 1: 28, 2: 1, 3: 1
candidates cuRobo could not plan, all trials: 11

## transfer, ground_truth (30 trials with a log)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 1 | 29 | 0 | 13.9 / 14.7 / 18.0 |
| 2 | 1 | 0 | 32.9 / 32.9 / 32.9 |

successful attempt planned with candidate: 1: 19, 2: 3, 3: 5, 4: 1, 6: 1, 8: 1
candidates cuRobo could not plan, all trials: 36

## transfer, perception (30 trials with a log)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 0 | 0 | 2 | -- |
| 1 | 27 | 1 | 14.3 / 14.7 / 16.8 |

successful attempt planned with candidate: 1: 22, 2: 3, 4: 2
candidates cuRobo could not plan, all trials: 9
