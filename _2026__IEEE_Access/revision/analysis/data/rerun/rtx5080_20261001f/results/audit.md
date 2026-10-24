# Re-run audit: `rtx5080_20261001f`

commit `63bb4b0254c64f917b80d22b8d4def613120ed49`, declared budgets 60,120,180 s

## Batches

| batch | attempts | last status | wall time (last) | max load avg (last) | scene |
|---|---|---|---|---|---|
| transfer_ground_truth_rep0 | 1 | done | 0.8 h | 9.3 | paper |
| move_ground_truth_rep0 | 1 | done | 0.6 h | 9.2 | paper |
| stir_ground_truth_rep0 | 1 | done | 0.9 h | 8.2 | paper |
| transfer_perception_rep0 | 2 | done | 1.2 h | 14.3 | paper |
| move_perception_rep0 | 3 | done | 0.0 h | 7.1 | paper |
| stir_perception_rep0 | 2 | done | 1.3 h | 14.3 | paper |
| pddlstream_transfer_ps0 | 2 | done | 0.4 h | 3.5 | paper |
| pddlstream_move_ps0 | 2 | done | 0.0 h | nan | paper |
| pddlstream_stir_ps0 | 2 | done | 0.0 h | 2.1 | paper |
| llm_xdl_generator | 1 | done | 0.0 h | 2.0 | paper |
| llm_ar_heldout_split2400 | 1 | done | 0.0 h | 2.1 | paper |
| llm_ar_unseen_control_in_distribution | 1 | done | 0.0 h | 2.2 | paper |
| llm_ar_unseen_L1_unseen_class | 1 | done | 0.0 h | 2.2 | paper |
| llm_ar_unseen_L2_unseen_layout | 1 | done | 0.0 h | nan | paper |
| llm_ar_unseen_L3_unseen_pair | 1 | done | 0.0 h | 2.1 | paper |
| pddlstream_perception_transfer_ps0 | 1 | done | 0.5 h | 3.0 | paper |
| pddlstream_perception_move_ps0 | 1 | done | 0.0 h | 1.2 | paper |
| pddlstream_perception_stir_ps0 | 1 | done | 0.0 h | 1.2 | paper |

## Trials: completeness and GPU exclusivity

| file | planner seed | layouts | duplicates | missing | trials that shared the GPU |
|---|---|---|---|---|---|
| `cutamp/move_ground_truth_rep0.csv` | - | 30 | - | - | - |
| `cutamp/move_perception_rep0.csv` | - | 30 | - | - | - |
| `cutamp/stir_ground_truth_rep0.csv` | - | 30 | - | - | - |
| `cutamp/stir_perception_rep0.csv` | - | 30 | - | - | - |
| `cutamp/transfer_ground_truth_rep0.csv` | - | 30 | - | - | - |
| `cutamp/transfer_perception_rep0.csv` | - | 30 | - | - | - |
| `pddlstream/pddlstream_move_5streams.csv` | 0 | 30 | - | - | - |
| `pddlstream/pddlstream_stir_5streams.csv` | 0 | 30 | - | - | - |
| `pddlstream/pddlstream_transfer_5streams.csv` | 0 | 30 | - | - | - |
| `pddlstream_perception/pddlstream_move_5streams.csv` | 0 | 30 | - | - | - |
| `pddlstream_perception/pddlstream_stir_5streams.csv` | 0 | 30 | - | - | - |
| `pddlstream_perception/pddlstream_transfer_5streams.csv` | 0 | 30 | - | - | - |

**Verdict: CLEAN**
