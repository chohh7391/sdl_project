# Re-run audit: `rtx5080_20261001e`

commit `580c948d2febbcbcef4d594105cab4c560387039`, declared budgets 60,120,180 s

## Batches

| batch | attempts | last status | wall time (last) | max load avg (last) | scene |
|---|---|---|---|---|---|
| transfer_ground_truth_rep0 | 1 | done | 0.8 h | 9.0 | paper |
| move_ground_truth_rep0 | 1 | done | 0.6 h | 10.0 | paper |
| stir_ground_truth_rep0 | 1 | done | 0.9 h | 10.4 | paper |
| transfer_perception_rep0 | 1 | done | 1.1 h | 12.8 | paper |
| pddlstream_transfer_ps0 | 1 | done | 0.2 h | 4.2 | paper |
| pddlstream_move_ps0 | 1 | done | 0.0 h | nan | paper |
| pddlstream_stir_ps0 | 1 | done | 0.0 h | 1.0 | paper |
| llm_xdl_generator | 1 | done | 0.0 h | 1.1 | paper |
| llm_ar_heldout_split2400 | 1 | done | 0.0 h | 1.0 | paper |
| llm_ar_unseen_control_in_distribution | 1 | done | 0.0 h | 0.9 | paper |
| llm_ar_unseen_L1_unseen_class | 1 | done | 0.0 h | 1.1 | paper |
| llm_ar_unseen_L2_unseen_layout | 1 | done | 0.0 h | 1.1 | paper |
| llm_ar_unseen_L3_unseen_pair | 1 | done | 0.0 h | 1.1 | paper |

## Trials: completeness and GPU exclusivity

| file | planner seed | layouts | duplicates | missing | trials that shared the GPU |
|---|---|---|---|---|---|
| `move_ground_truth_rep0.csv` | - | 30 | - | - | - |
| `stir_ground_truth_rep0.csv` | - | 30 | - | - | - |
| `transfer_ground_truth_rep0.csv` | - | 30 | - | - | - |
| `transfer_perception_rep0.csv` | - | 30 | - | - | - |
| `pddlstream_move_5streams.csv` | 0 | 30 | - | - | - |
| `pddlstream_stir_5streams.csv` | 0 | 30 | - | - | - |
| `pddlstream_transfer_5streams.csv` | 0 | 30 | - | - | - |

**Verdict: CLEAN**
