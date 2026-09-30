# Perception state: localization, error, the check before execution, success


## move (30 perception trials)

| vessel | fixed cameras | wrist recovery | not localized | localization with recovery [Wilson 95%] |
|---|---|---|---|---|
| beaker | 29 | 0 | 1 | 29/30 = 96.7% [83.3, 99.4] |
| flask | 29 | 0 | 0 | 29/29 = 100.0% [88.3, 100.0] |

trials that needed the wrist scan: 1; time localizing in them [s] median / max: 40.1 / 40.1

Pose error against the simulator, median / mean / p90 / max:

| vessel | source | horizontal [mm] | height, absolute [mm] | yaw, absolute [deg] |
|---|---|---|---|---|
| beaker | fixed | 2.8 / 2.8 / 3.9 / 6.8 (n=29) | 3.1 / 3.4 / 4.8 / 8.2 (n=29) | 0.1 / 0.1 / 0.2 / 0.2 (n=29) |
| flask | fixed | 2.2 / 2.5 / 3.7 / 5.6 (n=29) | 4.3 / 4.6 / 6.8 / 7.9 (n=29) | 0.1 / 0.1 / 0.1 / 0.2 (n=29) |

The check before execution (fixed cameras, after planning):

| vessel | verified | moved (planned again) | unseen | disagreement when seen [mm] median / max |
|---|---|---|---|---|
| beaker | 29 | 0 | 0 | 0.1 / 1.2 |
| flask | 29 | 0 | 0 | 0.0 / 1.0 |

plans redone after a vessel moved: 0

Task success, ground-truth vs perception state, same layouts:

| repetition | ground truth | perception | ground truth only | perception only | exact McNemar p |
|---|---|---|---|---|---|
| 0 | 30/30 | 29/30 | 1 | 0 | 1 |

pooled: ground truth 30/30 = 100.0% [88.6, 100.0], perception 29/30 = 96.7% [83.3, 99.4]
perception, every vessel from the fixed cameras: 29/29 = 100.0% [88.3, 100.0]
perception, a vessel recovered by the wrist scan: 0/0

## stir (30 perception trials)

| vessel | fixed cameras | wrist recovery | not localized | localization with recovery [Wilson 95%] |
|---|---|---|---|---|
| beaker | 29 | 1 | 0 | 30/30 = 100.0% [88.6, 100.0] |
| flask | 30 | 0 | 0 | 30/30 = 100.0% [88.6, 100.0] |

trials that needed the wrist scan: 1; time localizing in them [s] median / max: 19.3 / 19.3

Pose error against the simulator, median / mean / p90 / max:

| vessel | source | horizontal [mm] | height, absolute [mm] | yaw, absolute [deg] |
|---|---|---|---|---|
| beaker | fixed | 2.8 / 2.9 / 4.1 / 5.2 (n=29) | 3.3 / 3.4 / 6.1 / 7.8 (n=29) | 0.1 / 0.1 / 0.2 / 0.2 (n=29) |
| beaker | recovery | 4.2 / 4.2 / 4.2 / 4.2 (n=1) | 4.4 / 4.4 / 4.4 / 4.4 (n=1) | 0.1 / 0.1 / 0.1 / 0.1 (n=1) |
| flask | fixed | 2.4 / 2.4 / 3.5 / 5.9 (n=30) | 4.5 / 4.7 / 7.3 / 9.2 (n=30) | 0.0 / 0.1 / 0.1 / 0.2 (n=30) |

The check before execution (fixed cameras, after planning):

| vessel | verified | moved (planned again) | unseen | disagreement when seen [mm] median / max |
|---|---|---|---|---|
| beaker | 29 | 0 | 1 | 0.2 / 0.9 |
| flask | 30 | 0 | 0 | 0.1 / 1.4 |

plans redone after a vessel moved: 0

Task success, ground-truth vs perception state, same layouts:

| repetition | ground truth | perception | ground truth only | perception only | exact McNemar p |
|---|---|---|---|---|---|
| 0 | 30/30 | 30/30 | 0 | 0 | 1 |

pooled: ground truth 30/30 = 100.0% [88.6, 100.0], perception 30/30 = 100.0% [88.6, 100.0]
perception, every vessel from the fixed cameras: 29/29 = 100.0% [88.3, 100.0]
perception, a vessel recovered by the wrist scan: 1/1 = 100.0% [20.7, 100.0]

## transfer (30 perception trials)

| vessel | fixed cameras | wrist recovery | not localized | localization with recovery [Wilson 95%] |
|---|---|---|---|---|
| beaker | 29 | 1 | 0 | 30/30 = 100.0% [88.6, 100.0] |
| flask | 30 | 0 | 0 | 30/30 = 100.0% [88.6, 100.0] |

trials that needed the wrist scan: 1; time localizing in them [s] median / max: 19.9 / 19.9

Pose error against the simulator, median / mean / p90 / max:

| vessel | source | horizontal [mm] | height, absolute [mm] | yaw, absolute [deg] |
|---|---|---|---|---|
| beaker | fixed | 2.6 / 5.5 / 4.5 / 77.0 (n=29) | 3.9 / 3.9 / 6.3 / 8.5 (n=29) | 0.1 / 0.2 / 0.2 / 4.0 (n=29) |
| beaker | recovery | 4.3 / 4.3 / 4.3 / 4.3 (n=1) | 4.6 / 4.6 / 4.6 / 4.6 (n=1) | 0.0 / 0.0 / 0.0 / 0.0 (n=1) |
| flask | fixed | 2.4 / 2.4 / 3.3 / 6.2 (n=30) | 4.9 / 4.9 / 7.2 / 9.6 (n=30) | 0.0 / 0.1 / 0.1 / 0.2 (n=30) |

The check before execution (fixed cameras, after planning):

| vessel | verified | moved (planned again) | unseen | disagreement when seen [mm] median / max |
|---|---|---|---|---|
| beaker | 28 | 0 | 1 | 0.2 / 0.5 |
| flask | 29 | 0 | 0 | 0.2 / 0.8 |

plans redone after a vessel moved: 0

Task success, ground-truth vs perception state, same layouts:

| repetition | ground truth | perception | ground truth only | perception only | exact McNemar p |
|---|---|---|---|---|---|
| 0 | 30/30 | 27/30 | 3 | 0 | 0.25 |

pooled: ground truth 30/30 = 100.0% [88.6, 100.0], perception 27/30 = 90.0% [74.4, 96.5]
perception, every vessel from the fixed cameras: 26/29 = 89.7% [73.6, 96.4]
perception, a vessel recovered by the wrist scan: 1/1 = 100.0% [20.7, 100.0]
