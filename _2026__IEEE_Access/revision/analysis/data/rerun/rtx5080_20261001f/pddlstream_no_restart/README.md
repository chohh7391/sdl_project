# PDDLStream without restarts, same tag

What `run_all.sh` measured at the tag commit (`COMMIT`, 63bb4b0): the runner as
it was then, one `solve()` per layout and no restart (`RERUN_PDDL_RESTART=0` now
reproduces that policy). Moved here from `pddlstream/` on 2026-09-30 17:35.
`pddlstream/` then took the run with restarts at `PDDL_COMMIT` (see
`RUN_INFO.md`, last section), and `RESULTS.md` was regenerated from that run.

- `pddlstream_<task>_5streams.csv`: the rows, unchanged.
- `planner.md`: the planner comparison `90_analyse.sh` wrote against these rows
  before the move. The "PDDLStream:" path in its header names `pddlstream/`,
  which is where the rows were then.
- `attempts_cutamp.md`: the cuTAMP attempt report from the same run. It is
  unchanged by the move.

This is reference material, not a result: the comparison the tag reports is
against `pddlstream/`.
