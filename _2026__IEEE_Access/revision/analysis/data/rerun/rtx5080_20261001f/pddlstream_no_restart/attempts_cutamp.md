# cuTAMP planning calls: restarts and cuRobo candidates

Per trial, from its tamp_server log. `attempts` = optimizations run; `candidate` = which satisfying particle cuRobo planned in the successful attempt (1 = the best, the only one tried before candidates were added). Times are the driver's wall clock, solved trials only -- success rates and censored times are in the planner comparison.


## move, ground_truth (30 trials with a log)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 1 | 30 | 0 | 8.8 / 8.9 / 9.0 |

successful attempt planned with candidate: 1: 30
candidates cuRobo could not plan, all trials: 0

## move, perception (30 trials with a log)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 0 | 0 | 1 | -- |
| 1 | 29 | 0 | 9.2 / 9.4 / 9.6 |

successful attempt planned with candidate: 1: 29
candidates cuRobo could not plan, all trials: 0

## stir, ground_truth (30 trials with a log)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 1 | 28 | 0 | 11.6 / 12.0 / 12.5 |
| 2 | 2 | 0 | 19.1 / 23.7 / 28.3 |

successful attempt planned with candidate: 1: 27, 2: 3
candidates cuRobo could not plan, all trials: 11

## stir, perception (30 trials with a log)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 1 | 28 | 0 | 12.2 / 12.5 / 13.9 |
| 2 | 2 | 0 | 19.9 / 27.4 / 34.9 |

successful attempt planned with candidate: 1: 28, 4: 1, 6: 1
candidates cuRobo could not plan, all trials: 16

## transfer, ground_truth (30 trials with a log)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 1 | 28 | 0 | 14.1 / 14.6 / 16.3 |
| 2 | 2 | 0 | 31.8 / 32.8 / 33.7 |

successful attempt planned with candidate: 1: 19, 2: 6, 3: 1, 4: 4
candidates cuRobo could not plan, all trials: 36

## transfer, perception (30 trials with a log)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 1 | 27 | 0 | 14.5 / 15.1 / 19.9 |
| 2 | 2 | 0 | 33.5 / 34.0 / 34.6 |
| 12 | 0 | 1 | -- |

successful attempt planned with candidate: 1: 20, 2: 3, 3: 1, 4: 2, 6: 1, 8: 2
candidates cuRobo could not plan, all trials: 142
