# SEAM Experiment Metric Summary Table

| Policy | Topology | Poisoning | Score (Mean ± 95% CI) | Self-BLEU (Mean ± 95% CI) | Contamination Rate (Mean ± 95% CI) | Runs |
|---|---|---|---|---|---|---|
| naive_overwrite | full_broadcast | clean | 1.0000 [1.0000, 1.0000] | 0.9916 [0.9725, 1.0000] | 0.00% [0.00%, 0.00%] | 3 |
| naive_overwrite | full_broadcast | internal | 0.6825 [0.0000, 1.0000] | 0.9983 [0.9911, 1.0000] | 0.00% [0.00%, 0.00%] | 3 |
| naive_overwrite | off | clean | 1.0000 [1.0000, 1.0000] | 0.9966 [0.9894, 1.0000] | 0.00% [0.00%, 0.00%] | 3 |
| naive_overwrite | off | internal | 0.6825 [0.0000, 1.0000] | 0.9983 [0.9911, 1.0000] | 0.00% [0.00%, 0.00%] | 3 |
| naive_overwrite | ring | clean | 1.0000 [1.0000, 1.0000] | 0.9954 [0.9828, 1.0000] | 0.00% [0.00%, 0.00%] | 3 |
| naive_overwrite | ring | internal | 0.6825 [0.0000, 1.0000] | 0.9937 [0.9854, 1.0000] | 0.00% [0.00%, 0.00%] | 3 |
| raw_trajectory_buffer | full_broadcast | clean | 1.0000 [1.0000, 1.0000] | 0.9990 [0.9983, 0.9997] | 0.00% [0.00%, 0.00%] | 3 |
| raw_trajectory_buffer | full_broadcast | internal | 0.6840 [0.0000, 1.0000] | 0.9987 [0.9968, 1.0000] | 0.00% [0.00%, 0.00%] | 3 |
| raw_trajectory_buffer | off | clean | 0.6941 [0.0000, 1.0000] | 0.9987 [0.9983, 0.9992] | 0.00% [0.00%, 0.00%] | 3 |
| raw_trajectory_buffer | off | internal | 0.3736 [0.0000, 1.0000] | 0.9985 [0.9974, 0.9996] | 0.00% [0.00%, 0.00%] | 3 |
| raw_trajectory_buffer | ring | clean | 1.0000 [1.0000, 1.0000] | 0.9991 [0.9985, 0.9996] | 0.00% [0.00%, 0.00%] | 3 |
| raw_trajectory_buffer | ring | internal | 0.6847 [0.0000, 1.0000] | 0.9987 [0.9967, 1.0000] | 0.00% [0.00%, 0.00%] | 3 |
| structured_incremental | full_broadcast | clean | 0.0498 [0.0405, 0.0590] | 0.9976 [0.9976, 0.9976] | 0.00% [0.00%, 0.00%] | 3 |
| structured_incremental | full_broadcast | internal | 0.0462 [0.0422, 0.0502] | 0.9979 [0.9977, 0.9980] | 0.00% [0.00%, 0.00%] | 3 |
| structured_incremental | off | clean | 0.0404 [0.0141, 0.0666] | 0.9976 [0.9976, 0.9976] | 0.00% [0.00%, 0.00%] | 3 |
| structured_incremental | off | internal | 0.0329 [0.0189, 0.0468] | 0.9978 [0.9976, 0.9980] | 0.00% [0.00%, 0.00%] | 3 |
| structured_incremental | ring | clean | 0.3651 [0.0000, 1.0000] | 0.9976 [0.9976, 0.9976] | 0.00% [0.00%, 0.00%] | 3 |
| structured_incremental | ring | internal | 0.0332 [0.0000, 0.0674] | 0.9978 [0.9975, 0.9981] | 0.00% [0.00%, 0.00%] | 3 |
