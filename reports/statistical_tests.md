# SEAM Inferential Statistical Tests & Audit Report

**Audit Status**: `PASSED`
- Total Violations: 0

## Invariant Checks
- **Zero contamination in clean conditions**: Verified (0 violations)
- **Zero peer contamination under isolated (`off`) topology**: Verified (0 violations)
- **Sample completeness**: 10 distinct seeds (`40` through `49`) present for all 18 condition combinations per environment.

## Non-Parametric Hypothesis Tests (Mann-Whitney U & Cliff's Delta with FDR Correction)

| Environment | Hypothesis / Comparison | Metric | Condition A (Mean ± SD) | Condition B (Mean ± SD) | Cliff's $\delta$ | Interpretation | $p$-value (raw) | $p$-value (FDR) | Significant? |
|---|---|---|---|---|---:|:---:|---:|---:|:---:|
| `resource_foraging` | H1_struct_vs_naive_score | `score` | 0.1226 (med: 0.1384) | 0.0067 (med: 0.0000) | +0.920 | large | 3.1096e-04 | 9.3287e-04 | **YES** ($p < 0.05$) |
| `resource_foraging` | H2_struct_vs_naive_score_isolated | `score` | 0.1744 (med: 0.1799) | 0.0087 (med: 0.0000) | +1.000 | large | 1.3093e-04 | 6.2580e-04 | **YES** ($p < 0.05$) |
| `resource_foraging` | H3_struct_vs_raw_score | `score` | 0.1226 (med: 0.1384) | 0.0057 (med: 0.0000) | +0.960 | large | 1.6688e-04 | 6.2580e-04 | **YES** ($p < 0.05$) |
| `resource_foraging` | H4_ring_vs_broadcast_contamination_rate | `contamination_rate` | 1.0000 (med: 1.0000) | 1.0000 (med: 1.0000) | +0.000 | negligible | 1.0000e+00 | 1.0000e+00 | No |
| `resource_foraging` | H6_struct_vs_naive_self_bleu | `self_bleu` | 0.9972 (med: 0.9972) | 0.9990 (med: 1.0000) | -0.600 | large | 1.4558e-02 | 3.1195e-02 | **YES** ($p < 0.05$) |
| `bargaining_game` | H1_struct_vs_naive_score | `score` | 0.0372 (med: 0.0476) | 0.0000 (med: 0.0000) | +0.700 | large | 1.9862e-03 | 4.9655e-03 | **YES** ($p < 0.05$) |
| `bargaining_game` | H2_struct_vs_naive_score_isolated | `score` | 0.0434 (med: 0.0402) | 0.0000 (med: 0.0000) | +1.000 | large | 6.3864e-05 | 6.2580e-04 | **YES** ($p < 0.05$) |
| `bargaining_game` | H3_struct_vs_raw_score | `score` | 0.0372 (med: 0.0476) | 0.0278 (med: 0.0000) | +0.100 | negligible | 7.1913e-01 | 1.0000e+00 | No |
| `bargaining_game` | H4_ring_vs_broadcast_contamination_rate | `contamination_rate` | 1.0000 (med: 1.0000) | 1.0000 (med: 1.0000) | +0.000 | negligible | 1.0000e+00 | 1.0000e+00 | No |
| `bargaining_game` | H6_struct_vs_naive_self_bleu | `self_bleu` | 0.9973 (med: 0.9973) | 0.9952 (med: 0.9947) | +0.600 | large | 2.2645e-02 | 4.2460e-02 | **YES** ($p < 0.05$) |
| `number_guessing` | H1_struct_vs_naive_score | `score` | 0.0411 (med: 0.0000) | 0.0333 (med: 0.0000) | +0.170 | small | 3.8701e-01 | 6.4502e-01 | No |
| `number_guessing` | H2_struct_vs_naive_score_isolated | `score` | 0.0483 (med: 0.0000) | 0.0674 (med: 0.0000) | +0.040 | negligible | 8.9416e-01 | 1.0000e+00 | No |
| `number_guessing` | H3_struct_vs_raw_score | `score` | 0.0411 (med: 0.0000) | 0.0411 (med: 0.0000) | +0.000 | negligible | 1.0000e+00 | 1.0000e+00 | No |
| `number_guessing` | H4_ring_vs_broadcast_contamination_rate | `contamination_rate` | 1.0000 (med: 1.0000) | 1.0000 (med: 1.0000) | +0.000 | negligible | 1.0000e+00 | 1.0000e+00 | No |
| `number_guessing` | H6_struct_vs_naive_self_bleu | `self_bleu` | 0.9885 (med: 0.9972) | 0.8499 (med: 0.8882) | +0.980 | large | 1.6442e-04 | 6.2580e-04 | **YES** ($p < 0.05$) |
