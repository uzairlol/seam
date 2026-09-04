# Metrics and Statistics

## Implemented metrics

### Final task score

The environment's `get_ground_truth_score()` result. Semantics are scenario-specific; see [scenarios](scenarios.md).

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

## Aggregation

`ResultAggregator.aggregate_conditions()` groups run-level rows by available `policy`, `topology`, and `poisoning_mode`. For each metric it computes:

- mean;
- sample standard deviation;
- count;
- SEM = standard deviation / sqrt(count);
- 95% t-distribution confidence interval, clipped to `[0, 1]`.

The generated Markdown table reports mean and confidence interval. It does not report raw per-seed values.

## Statistical cautions

- Three seeds are exploratory; five is a better minimum, while 10-20 is more defensible for manuscript claims.
- Confidence intervals with small `n` are wide and should not be treated as proof of no effect.
- Do not compare metrics across environments as if they share one objective scale.
- Confirm equal condition counts before aggregation.
- Inspect run-level distributions, not only bar-chart means.
- A contamination association is not automatically a causal performance effect.

## Proposed but not currently emitted

Project-scope documents also discuss propagation latency, semantic poison similarity, regret, inter-agent reward variance, and per-round contamination trajectories. These require implementation and validation before appearing as empirical claims.
