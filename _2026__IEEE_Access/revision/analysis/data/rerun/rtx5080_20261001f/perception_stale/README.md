# Perception batches from before the rebuild fix, same tag

The three perception batches `run_all.sh` measured at the tag commit
(`COMMIT`, 63bb4b0), moved here from `cutamp/` and `logs/` on 2026-09-30
18:44. They are not results: two defects put the scene before each trial's
tool change into the World State.

- **Pre-rebuild observations.** 34 of these 90 trials planned from an
  observation stamped before the simulator rebuilt its world. Delivered after
  the TF buffers had been cleared, it passed the age test, which only rejected
  old observations. Most were 1-5 mm off, because the vessels had not moved;
  Transfer seed 24's beaker was 77 mm off, and that trial failed.
- **Wrist camera without an arm pose.** In Move (vgc10), robot_state_publisher
  stopped publishing the arm for ~50 s after the rebuild. The wrist scan
  therefore found nothing, and Move seed 17 failed.

Both are fixed in a50f925. The batches were then measured again into
`cutamp/` at that commit (`PERCEPTION_COMMIT`); the ground-truth batches were
not re-run.

- `cutamp/`, `logs/`: the rows and per-trial logs, unchanged.
- `results/`: the perception and state-source reports `90_analyse.sh` wrote
  from them.
