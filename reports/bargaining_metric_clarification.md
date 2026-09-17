# Game-Theoretic Clarification of Bargaining Metrics

## Motivation & Formal Distinction
In multi-agent bargaining literature, a common source of reviewer confusion is conflating **Fairness** with **Efficiency (Welfare)** or **Agreement Frequency (Acceptance Rate)**.

In SEAM's Bargaining Environment:
1. **Fairness Score** ($S_{\text{fair}}$):
   $$S_{\text{fair}} = \frac{1}{1 + \frac{1}{|\mathcal{D}|}\sum_{(s_1, s_2) \in \mathcal{D}} |s_1 - 50|}$$
   where $\mathcal{D}$ is the set of accepted deals (pie size = 100). If no deal is struck ($|\mathcal{D}| = 0$), $S_{\text{fair}} = 0.0$.
2. **Acceptance Rate** ($\alpha$):
   $$\alpha = \frac{|\mathcal{D}|}{T}$$
   Measuring negotiation deadlock vs. convergence.
3. **Total Welfare** ($W$):
   $$W = \sum_{(s_1, s_2) \in \mathcal{D}} (s_1 + s_2) = 100 \times |\mathcal{D}|$$

### Key Insight:
An accepted proposal of `99/1` achieves maximal welfare ($100$ per round) but exhibits minimal fairness ($S_{\text{fair}} = 1/(1+49) = 0.02$). Conversely, reaching a single `50/50` agreement across 20 rounds achieves perfect conditional fairness ($1.0$) but low total welfare. Therefore, SEAM reports **Fairness Score** to measure whether self-evolving reflection helps agents converge on the egalitarian Nash bargaining solution rather than greedy unilateral exploitation.

## Empirical Comparison Across Policies & Topologies

| Policy | Topology | Poisoning | Mean Fairness Score | Mean Acceptance Rate | Mean Total Welfare | Runs ($N$) |
|---|---|---|---:|---:|---:|---:|
| `naive_overwrite` | `full_broadcast` | `clean` | 0.0048 ± 0.015 | 0.5% ± 1.6% | 10.0 | 10 |
| `naive_overwrite` | `full_broadcast` | `internal` | 0.0196 ± 0.000 | 100.0% ± 0.0% | 2000.0 | 10 |
| `naive_overwrite` | `off` | `clean` | 0.0000 ± 0.000 | 0.0% ± 0.0% | 0.0 | 10 |
| `naive_overwrite` | `off` | `internal` | 0.0097 ± 0.021 | 14.5% ± 31.7% | 290.0 | 10 |
| `naive_overwrite` | `ring` | `clean` | 0.0000 ± 0.000 | 0.0% ± 0.0% | 0.0 | 10 |
| `naive_overwrite` | `ring` | `internal` | 0.0253 ± 0.003 | 88.0% ± 15.5% | 1760.0 | 10 |
| `raw_trajectory_buffer` | `full_broadcast` | `clean` | 0.0308 ± 0.041 | 31.5% ± 42.8% | 630.0 | 10 |
| `raw_trajectory_buffer` | `full_broadcast` | `internal` | 0.0369 ± 0.007 | 92.5% ± 17.4% | 1850.0 | 10 |
| `raw_trajectory_buffer` | `off` | `clean` | 0.0251 ± 0.033 | 26.0% ± 35.8% | 520.0 | 10 |
| `raw_trajectory_buffer` | `off` | `internal` | 0.0186 ± 0.024 | 24.0% ± 32.5% | 480.0 | 10 |
| `raw_trajectory_buffer` | `ring` | `clean` | 0.0278 ± 0.037 | 26.5% ± 34.6% | 530.0 | 10 |
| `raw_trajectory_buffer` | `ring` | `internal` | 0.0413 ± 0.009 | 91.5% ± 20.1% | 1830.0 | 10 |
| `structured_incremental` | `full_broadcast` | `clean` | 0.0275 ± 0.030 | 42.0% ± 44.4% | 840.0 | 10 |
| `structured_incremental` | `full_broadcast` | `internal` | 0.0197 ± 0.000 | 100.0% ± 0.0% | 2000.0 | 10 |
| `structured_incremental` | `off` | `clean` | 0.0434 ± 0.009 | 79.5% ± 15.9% | 1590.0 | 10 |
| `structured_incremental` | `off` | `internal` | 0.0418 ± 0.011 | 89.0% ± 12.0% | 1780.0 | 10 |
| `structured_incremental` | `ring` | `clean` | 0.0372 ± 0.027 | 63.0% ± 43.6% | 1260.0 | 10 |
| `structured_incremental` | `ring` | `internal` | 0.0257 ± 0.007 | 99.0% ± 3.2% | 1980.0 | 10 |

## Recommendations for Manuscript Writing
- Always refer to this metric as **Fairness Score** (or *Egalitarian Fairness Index*), never generic 'performance' or 'reward'.
- Note in the Methods section that when negotiations completely deadlock ($0$ agreements), Fairness Score is safely bounded at $0.0$.
- Explicitly contrast with Resource Foraging (which measures environmental extraction efficiency) and Number Guessing (which measures coordination speed $1/T$).
