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
| 1 | 30 | 0 | 9.2 / 9.3 / 10.4 |

successful attempt planned with candidate: 1: 30
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
| 1 | 29 | 0 | 12.1 / 12.5 / 20.8 |
| 2 | 1 | 0 | 19.7 / 19.7 / 19.7 |

successful attempt planned with candidate: 1: 27, 2: 1, 3: 1, 8: 1
candidates cuRobo could not plan, all trials: 10

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
| 1 | 29 | 0 | 14.5 / 15.1 / 18.3 |
| 2 | 1 | 0 | 33.2 / 33.2 / 33.2 |

successful attempt planned with candidate: 1: 18, 2: 8, 3: 1, 4: 2, 5: 1
candidates cuRobo could not plan, all trials: 28

# PDDLStream planning calls: restarts

Per trial, from its CSV. `attempts` = solve() calls within the limit; each restart puts the scene back and draws a fresh sample stream.


## pddlstream, move (30 trials, restart 1)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 1 | 30 | 0 | 0.0 / 0.1 / 0.6 |

solved by the first solve() alone: 30/30; solved only after a restart: 0
the same tag's run without restarts (pddlstream_no_restart/): 30/30 planned

## pddlstream, stir (30 trials, restart 1)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 1 | 20 | 0 | 0.3 / 0.5 / 1.3 |
| 2 | 5 | 0 | 1.5 / 2.3 / 3.8 |
| 3-9 | 5 | 0 | 3.5 / 5.8 / 8.9 |

solved by the first solve() alone: 20/30; solved only after a restart: 10
the same tag's run without restarts (pddlstream_no_restart/): 23/30 planned

## pddlstream, transfer (30 trials, restart 1)

| attempts | planned | not planned | solved-only time min / median / max [s] |
|---|---|---|---|
| 1 | 20 | 7 | 0.3 / 1.0 / 59.5 |
| 2 | 1 | 1 | 45.0 / 45.0 / 45.0 |
| 3-9 | 1 | 0 | 14.4 / 14.4 / 14.4 |

solved by the first solve() alone: 20/30; solved only after a restart: 2
not planned, by reason: no_plan_within_budget 8
the same tag's run without restarts (pddlstream_no_restart/): 20/30 planned
