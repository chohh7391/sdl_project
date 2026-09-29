# Re-run audit: `rtx5080_20261001b`

commit `e1ce911acf1d0dfb755c44c451d10af59e404423`, declared budgets 60,120,180 s

## Batches

| batch | attempts | last status | wall time (last) | max load avg (last) | scene |
|---|---|---|---|---|---|
| transfer_ground_truth_rep0 | 2 | done | 0.0 h | 4.5 | paper |
| move_ground_truth_rep0 | 2 | done | 0.0 h | 7.9 | paper |
| stir_ground_truth_rep0 | 2 | done | 0.1 h | 8.0 | paper |
| transfer_ground_truth_rep1 | 1 | done | 0.8 h | 9.6 | paper |
| move_ground_truth_rep1 | 1 | done | 0.6 h | 8.4 | paper |
| stir_ground_truth_rep1 | 1 | done | 0.8 h | 9.7 | paper |
| transfer_ground_truth_rep2 | 1 | done | 0.8 h | 8.3 | paper |
| move_ground_truth_rep2 | 1 | done | 0.6 h | 9.3 | paper |
| stir_ground_truth_rep2 | 1 | done | 0.8 h | 9.5 | paper |
| transfer_perception_rep0 | 2 | done | 0.0 h | 10.5 | paper |
| transfer_perception_rep1 | 2 | done | 0.0 h | 11.9 | paper |
| transfer_perception_rep2 | 2 | done | 0.0 h | 7.4 | paper |
| pddlstream_transfer_ps0 | 2 | done | 0.4 h | 3.1 | paper |
| pddlstream_transfer_ps1 | 2 | done | 0.2 h | 1.1 | paper |
| pddlstream_transfer_ps2 | 2 | done | 0.3 h | 1.1 | paper |
| pddlstream_transfer_ps3 | 2 | done | 0.2 h | 2.9 | paper |
| pddlstream_transfer_ps4 | 2 | done | 0.3 h | 2.1 | paper |
| pddlstream_move_ps0 | 2 | done | 0.0 h | nan | paper |
| pddlstream_move_ps1 | 2 | done | 0.0 h | 1.0 | paper |
| pddlstream_move_ps2 | 2 | done | 0.0 h | nan | paper |
| pddlstream_move_ps3 | 2 | done | 0.0 h | nan | paper |
| pddlstream_move_ps4 | 2 | done | 0.0 h | nan | paper |
| pddlstream_stir_ps0 | 2 | done | 0.0 h | 1.0 | paper |
| pddlstream_stir_ps1 | 2 | done | 0.0 h | 1.1 | paper |
| pddlstream_stir_ps2 | 2 | done | 0.0 h | 1.0 | paper |
| pddlstream_stir_ps3 | 2 | done | 0.0 h | 1.1 | paper |
| pddlstream_stir_ps4 | 2 | done | 0.0 h | 1.1 | paper |
| llm_xdl_generator | 1 | done | 0.1 h | 7.5 | paper |
| llm_ar_heldout_split2400 | 1 | done | 0.0 h | 1.3 | paper |
| llm_ar_unseen_control_in_distribution | 1 | done | 0.0 h | 0.9 | paper |
| llm_ar_unseen_L1_unseen_class | 1 | done | 0.0 h | 1.0 | paper |
| llm_ar_unseen_L2_unseen_layout | 1 | done | 0.0 h | nan | paper |
| llm_ar_unseen_L3_unseen_pair | 1 | done | 0.0 h | 1.1 | paper |

## Trials: completeness and GPU exclusivity

| file | planner seed | layouts | duplicates | missing | trials that shared the GPU |
|---|---|---|---|---|---|
| `move_ground_truth_rep0.csv` | - | 30 | - | - | - |
| `move_ground_truth_rep1.csv` | - | 30 | - | - | - |
| `move_ground_truth_rep2.csv` | - | 30 | - | - | - |
| `stir_ground_truth_rep0.csv` | - | 30 | - | - | - |
| `stir_ground_truth_rep1.csv` | - | 30 | - | - | - |
| `stir_ground_truth_rep2.csv` | - | 30 | - | - | - |
| `transfer_ground_truth_rep0.csv` | - | 30 | - | - | - |
| `transfer_ground_truth_rep1.csv` | - | 30 | - | - | - |
| `transfer_ground_truth_rep2.csv` | - | 30 | - | - | - |
| `transfer_perception_rep0.csv` | - | 30 | - | - | - |
| `transfer_perception_rep1.csv` | - | 30 | - | - | - |
| `transfer_perception_rep2.csv` | - | 30 | - | - | - |
| `pddlstream_move_5streams.csv` | 0 | 30 | - | - | - |
| `pddlstream_move_5streams.csv` | 1 | 30 | - | - | - |
| `pddlstream_move_5streams.csv` | 2 | 30 | - | - | - |
| `pddlstream_move_5streams.csv` | 3 | 30 | - | - | - |
| `pddlstream_move_5streams.csv` | 4 | 30 | - | - | - |
| `pddlstream_stir_5streams.csv` | 0 | 30 | - | - | - |
| `pddlstream_stir_5streams.csv` | 1 | 30 | - | - | - |
| `pddlstream_stir_5streams.csv` | 2 | 30 | - | - | - |
| `pddlstream_stir_5streams.csv` | 3 | 30 | - | - | - |
| `pddlstream_stir_5streams.csv` | 4 | 30 | - | - | - |
| `pddlstream_transfer_5streams.csv` | 0 | 30 | - | - | - |
| `pddlstream_transfer_5streams.csv` | 1 | 30 | - | - | - |
| `pddlstream_transfer_5streams.csv` | 2 | 30 | - | - | - |
| `pddlstream_transfer_5streams.csv` | 3 | 30 | - | - | - |
| `pddlstream_transfer_5streams.csv` | 4 | 30 | - | - | - |

**Verdict: CLEAN**
