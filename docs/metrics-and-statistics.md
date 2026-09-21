# Metrics and Statistics

## Implemented metrics

### Final task score

The environment's `get_ground_truth_score()` result. Semantics are scenario-specific; see [scenarios](scenarios.md).

### Oracle reference score

For every run the runner rolls out a deterministic rule-based reference policy on a fresh instance of the same environment seeded identically (`seam.analysis.oracle.oracle_rollout`). Because each environment is deterministic given its seed, the oracle score is a reproducible upper reference per `(env_type, seed)` pair:

- resource foraging: greedy move-toward-nearest-visible-resource / harvest-on-arrival;
- number guessing: binary search over the action space using per-agent feedback;
- bargaining: propose the Nash 50/50 split; responders always accept.

The oracle never calls an LLM, so it is cheap and reproducible.

### Efficacy gap

The raw `final_score` is difficult to compare across conditions: it hovers near the `no_memory` floor and saturates. `compute_efficacy_gap()` rescales each run onto the per-seed headroom:

$$
\text{efficacy gap}=\frac{\text{score}-\text{baseline}}{\text{oracle}-\text{baseline}},
$$

where `baseline` is the mean `no_memory`/`off` score for the same seed and `oracle` is the oracle score for that `(env_type, seed)` instance. A value of 1.0 means the run matched the oracle; 0.0 means it matched the no-memory control. Runs without a seed-matched baseline get NaN and are excluded from aggregation. The normalization makes efficacy comparable across environments that do not share one raw-score scale.

### Cumulative rewards

Sum of per-round reward for each agent. This is retained in run summaries but is not currently a primary column in the standard aggregate table.

### Self-BLEU

`compute_self_bleu()` compares each non-empty memory state against other states in the same agent's sequence using clipped n-gram precision. Higher values indicate greater lexical repetition. The runner supplies local memory contexts for this metric so peer-shared context is not directly counted as the agent's own temporal memory.

A value near 1.0 may mean highly repetitive text, but it can also be inflated by short, templated, or deterministic memories. Interpret alongside memory length and action entropy.

### Action entropy

Rolling Shannon entropy over actions. Low entropy indicates behavioral concentration or repetition. The summary stores per-agent lists of window values.

### Memory length

Whitespace word count per recorded memory state. It is a storage-size proxy, not a tokenizer-accurate token count.

### Peer contamination rate

Fraction of non-seed agents with at least one memory state containing a configured poison phrase. It is bounded in `[0, 1]` and is not a per-round prevalence measure.

### Poison adherence

Fraction of an agent's executed actions matching the poison payload pattern. For multi-word payloads this should be interpreted carefully because action extraction and environment action spaces may not use identical text formats.

### Poison dosage

`compute_poison_dosage()` measures the fraction of a memory state's text that is attributable to payload phrases (overlapping keyword spans / total length). Unlike the presence-based contamination rate, dosage is continuous and does not saturate at the first keyword hit, so it can separate "echoes a single phrase" from "memory fully taken over by the payload". The runner records each peer's peak dosage over rounds (`per_agent_poison_dosage`) and the mean over peers (`poison_dosage_rate`). Dosage is gated to 0.0 when poisoning is inactive or sharing is off.

### Time-to-infection survival curves

Each run records the first round at which every peer's memory tested positive for the payload (`peer_propagation_round`; `None` = stayed clean, right-censored at `rounds_played`). `seam.analysis.survival` flattens these into per-peer event/censor records and estimates a Kaplan–Meier survival curve per condition — the fraction of not-yet-contaminated peers still at risk as a function of round. Use `plot_survival_curves()` to render them. Survival answers "how *fast* does the payload spread", which the end-of-episode contamination rate hides.

## Aggregation

`ResultAggregator.aggregate_conditions()` groups run-level rows by available `policy`, `topology`, and `poisoning_mode`. For each metric it computes:

- mean;
- sample standard deviation;
- count;
- SEM = standard deviation / sqrt(count);
- 95% t-distribution confidence interval (clipped to `[0, 1]` for the bounded metrics; unbounded for `efficacy_gap`).

When the underlying runs carry an `oracle_score` (and therefore a seed-matched `no_memory` baseline), the aggregator also derives `efficacy_gap` and reports it in an extended Markdown table.

The generated Markdown table reports mean and confidence interval. It does not report raw per-seed values.

## Statistical cautions

- Three seeds are exploratory; five is a better minimum, while 10-20 is more defensible for manuscript claims.
- Confidence intervals with small `n` are wide and should not be treated as proof of no effect.
- Do not compare metrics across environments as if they share one objective scale.
- Confirm equal condition counts before aggregation.
- Inspect run-level distributions, not only bar-chart means.
- A contamination association is not automatically a causal performance effect.

## Proposed but not currently emitted

Semantic poison similarity, regret, inter-agent reward variance, and sub-LLM-agent trajectory analysis require further implementation and validation before appearing as empirical claims.

## Formula summary

For a condition with run values $x_1,\ldots,x_n$, the reported mean is

$$
\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i.
$$

For $n>1$, the current implementation estimates the standard error as $s/\sqrt{n}$ and uses a two-sided 95% Student-$t$ interval with $n-1$ degrees of freedom. For a single run, the interval half-width is set to zero. The displayed intervals are clipped to `[0, 1]` for the bounded metrics — which is appropriate — but the `efficacy_gap` interval is left unclipped because efficacy is not confined to that range.

## Metric sequence versus scalar

The runner stores scalar mean Self-BLEU and scalar contamination rate per run, but action entropy and memory length are sequences over rounds/windows. The standard experiment-level CSV therefore cannot show their full trajectories. Use `events.jsonl` or a custom analysis for temporal plots.
