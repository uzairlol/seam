# SEAM Experiment Metric Summary Table

| Policy | Topology | Poisoning | Score (Mean ± 95% CI) | Self-BLEU (Mean ± 95% CI) | Contamination Rate (Mean ± 95% CI) | Runs |
|---|---|---|---|---|---|---|
| naive_overwrite | broadcast | clean | 0.2513 [0.0797, 0.4228] | 0.9780 [0.9553, 1.0000] | 0.00% [0.00%, 0.00%] | 26 |
| naive_overwrite | broadcast | internal | 0.1714 [0.0243, 0.3185] | 0.9864 [0.9747, 0.9980] | 30.77% [11.76%, 49.78%] | 26 |
| naive_overwrite | full_broadcast | clean | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] | 0.00% [0.00%, 0.00%] | 1 |
| naive_overwrite | full_broadcast | internal | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] | 100.00% [100.00%, 100.00%] | 1 |
| naive_overwrite | off | clean | 0.3715 [0.1383, 0.6047] | 0.9354 [0.8783, 0.9925] | 0.00% [0.00%, 0.00%] | 18 |
| naive_overwrite | off | internal | 0.2720 [0.0000, 0.5993] | 0.9409 [0.8513, 1.0000] | 0.00% [0.00%, 0.00%] | 9 |
| raw_trajectory_buffer | broadcast | clean | 0.3641 [0.1333, 0.5949] | 0.9970 [0.9945, 0.9995] | 0.00% [0.00%, 0.00%] | 18 |
| raw_trajectory_buffer | broadcast | internal | 0.2743 [0.0681, 0.4805] | 0.9867 [0.9630, 1.0000] | 0.00% [0.00%, 0.00%] | 18 |
| raw_trajectory_buffer | off | clean | 0.2525 [0.0473, 0.4577] | 0.9969 [0.9944, 0.9993] | 0.00% [0.00%, 0.00%] | 18 |
| raw_trajectory_buffer | off | internal | 0.1446 [0.0000, 0.3928] | 0.9968 [0.9930, 1.0000] | 0.00% [0.00%, 0.00%] | 9 |
| structured_incremental | broadcast | clean | 0.1358 [0.0232, 0.2484] | 0.9965 [0.9950, 0.9980] | 0.00% [0.00%, 0.00%] | 18 |
| structured_incremental | broadcast | internal | 0.0424 [0.0141, 0.0706] | 0.9910 [0.9807, 1.0000] | 5.26% [0.00%, 16.32%] | 19 |
| structured_incremental | full_broadcast | internal | 0.0000 [0.0000, 0.0000] | 0.9089 [0.9089, 0.9089] | 100.00% [100.00%, 100.00%] | 1 |
| structured_incremental | off | clean | 0.0858 [0.0471, 0.1246] | 0.9965 [0.9951, 0.9980] | 0.00% [0.00%, 0.00%] | 18 |
| structured_incremental | off | internal | 0.0705 [0.0218, 0.1193] | 0.9968 [0.9945, 0.9991] | 0.00% [0.00%, 0.00%] | 9 |
