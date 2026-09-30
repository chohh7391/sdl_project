# Perception state: localization, error, the check before execution, success


## move (30 perception trials)

| vessel | fixed cameras | wrist recovery | not localized | localization with recovery [Wilson 95%] |
|---|---|---|---|---|
| beaker | 29 | 1 | 0 | 30/30 = 100.0% [88.6, 100.0] |
| flask | 30 | 0 | 0 | 30/30 = 100.0% [88.6, 100.0] |

trials that needed the wrist scan: 1; time localizing in them [s] median / max: 18.8 / 18.8

Pose error against the simulator, median / mean / p90 / max:

| vessel | source | horizontal [mm] | height, absolute [mm] | yaw, absolute [deg] |
|---|---|---|---|---|
| beaker | fixed | 2.6 / 3.0 / 4.4 / 6.9 (n=29) | 3.4 / 3.2 / 5.1 / 8.5 (n=29) | 0.1 / 0.1 / 0.2 / 0.2 (n=29) |
| beaker | recovery | 4.2 / 4.2 / 4.2 / 4.2 (n=1) | 4.5 / 4.5 / 4.5 / 4.5 (n=1) | 0.0 / 0.0 / 0.0 / 0.0 (n=1) |
| flask | fixed | 2.4 / 2.5 / 3.8 / 5.2 (n=30) | 5.2 / 5.0 / 7.5 / 9.0 (n=30) | 0.0 / 0.1 / 0.1 / 0.3 (n=30) |

The check before execution (fixed cameras, after planning):

| vessel | verified | moved (planned again) | unseen | disagreement when seen [mm] median / max |
|---|---|---|---|---|
| beaker | 29 | 0 | 1 | 0.3 / 1.2 |
| flask | 30 | 0 | 0 | 0.3 / 0.8 |

plans redone after a vessel moved: 0

Task success, ground-truth vs perception state, same layouts:

| repetition | ground truth | perception | ground truth only | perception only | exact McNemar p |
|---|---|---|---|---|---|
| 0 | 30/30 | 30/30 | 0 | 0 | 1 |

pooled: ground truth 30/30 = 100.0% [88.6, 100.0], perception 30/30 = 100.0% [88.6, 100.0]
perception, every vessel from the fixed cameras: 29/29 = 100.0% [88.3, 100.0]
perception, a vessel recovered by the wrist scan: 1/1 = 100.0% [20.7, 100.0]

## stir (30 perception trials)

| vessel | fixed cameras | wrist recovery | not localized | localization with recovery [Wilson 95%] |
|---|---|---|---|---|
| beaker | 29 | 1 | 0 | 30/30 = 100.0% [88.6, 100.0] |
| flask | 30 | 0 | 0 | 30/30 = 100.0% [88.6, 100.0] |

trials that needed the wrist scan: 1; time localizing in them [s] median / max: 19.2 / 19.2

Pose error against the simulator, median / mean / p90 / max:

| vessel | source | horizontal [mm] | height, absolute [mm] | yaw, absolute [deg] |
|---|---|---|---|---|
| beaker | fixed | 2.6 / 3.0 / 4.2 / 6.9 (n=29) | 3.0 / 3.1 / 5.1 / 8.5 (n=29) | 0.1 / 0.1 / 0.2 / 0.2 (n=29) |
| beaker | recovery | 4.4 / 4.4 / 4.4 / 4.4 (n=1) | 4.7 / 4.7 / 4.7 / 4.7 (n=1) | 0.1 / 0.1 / 0.1 / 0.1 (n=1) |
| flask | fixed | 2.2 / 2.5 / 3.3 / 5.7 (n=30) | 5.5 / 5.1 / 7.5 / 8.0 (n=30) | 0.1 / 0.1 / 0.1 / 0.3 (n=30) |

The check before execution (fixed cameras, after planning):

| vessel | verified | moved (planned again) | unseen | disagreement when seen [mm] median / max |
|---|---|---|---|---|
| beaker | 29 | 0 | 1 | 0.3 / 1.1 |
| flask | 30 | 0 | 0 | 0.4 / 1.0 |

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

trials that needed the wrist scan: 1; time localizing in them [s] median / max: 19.5 / 19.5

Pose error against the simulator, median / mean / p90 / max:

| vessel | source | horizontal [mm] | height, absolute [mm] | yaw, absolute [deg] |
|---|---|---|---|---|
| beaker | fixed | 3.0 / 3.1 / 4.2 / 6.9 (n=29) | 3.1 / 3.2 / 5.0 / 8.6 (n=29) | 0.1 / 0.1 / 0.2 / 0.2 (n=29) |
| beaker | recovery | 4.4 / 4.4 / 4.4 / 4.4 (n=1) | 4.7 / 4.7 / 4.7 / 4.7 (n=1) | 0.0 / 0.0 / 0.0 / 0.0 (n=1) |
| flask | fixed | 2.4 / 2.6 / 3.3 / 5.5 (n=30) | 5.6 / 5.3 / 7.5 / 8.6 (n=30) | 0.0 / 0.1 / 0.1 / 0.2 (n=30) |

The check before execution (fixed cameras, after planning):

| vessel | verified | moved (planned again) | unseen | disagreement when seen [mm] median / max |
|---|---|---|---|---|
| beaker | 29 | 0 | 1 | 0.3 / 1.1 |
| flask | 30 | 0 | 0 | 0.3 / 1.3 |

plans redone after a vessel moved: 0

Task success, ground-truth vs perception state, same layouts:

| repetition | ground truth | perception | ground truth only | perception only | exact McNemar p |
|---|---|---|---|---|---|
| 0 | 30/30 | 29/30 | 1 | 0 | 1 |

pooled: ground truth 30/30 = 100.0% [88.6, 100.0], perception 29/30 = 96.7% [83.3, 99.4]
perception, every vessel from the fixed cameras: 28/29 = 96.6% [82.8, 99.4]
perception, a vessel recovered by the wrist scan: 1/1 = 100.0% [20.7, 100.0]
