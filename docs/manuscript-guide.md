# Manuscript Guide

## Suggested paper structure

### 1. Introduction

Define self-evolving agent memory, identify collapse and contamination as separate risks, and motivate their interaction in multi-agent communication.

### 2. Related work

Organize the literature by mechanism: memory evolution, collapse/echo effects, experience poisoning, multi-agent communication, and benchmarks. Use [literature-map](literature-map.md) to avoid turning the section into an unstructured paper list.

### 3. Research questions and hypotheses

State which effects are predicted before showing results. Example dimensions include policy robustness, topology-dependent propagation, and whether sharing changes collapse independently of poisoning.

### 4. System and environments

Describe agents, Ollama model configuration, prompt construction, memory policies, communication topology, poison condition, and all three task scenarios. Keep fairness score in bargaining separate from generic performance.

### 5. Experimental protocol

Report the factorial design, number of agents, episode length, publish cadence, memory limits, poison payloads, model tag, decoding settings, and exact seed set. Include exclusion/incomplete-run rules.

### 6. Metrics and analysis

Define every metric mathematically or operationally, explain aggregation, and distinguish exploratory descriptive statistics from inferential claims. Include raw run counts and confidence intervals.

### 7. Results

Present within-environment comparisons first. Use performance, collapse, and contamination figures together. Explain surprising cases using event logs rather than inferring from bar heights alone.

### 8. Limitations

Discuss toy-task scope, one-model focus, deterministic decoding, manually designed payloads, phrase-based contamination detection, small seed counts, and the distinction between fairness and welfare in bargaining.

### 9. Conclusion

Return to the central question without claiming more than the design identifies.

## Evidence table to maintain

For each manuscript claim, record:

| Claim | Environment | Condition comparison | Seeds | Run artifact location | Metric | Caveat |
|---|---|---|---:|---|---|---|
| Example: ring limits spread relative to broadcast | resource_foraging | internal poisoning | 5+ | `runs/experiments/resource_foraging/` | peer contamination | descriptive unless tested statistically |

## Writing rules

- Never pool pre-fix and post-fix runs.
- Cite condition counts, not only aggregate means.
- Use “associated with” unless the design supports a causal statement.
- Explain near-1 Self-BLEU values with memory length and action entropy.
- Report failed or missing runs transparently.
- Separate implementation facts from hypotheses and future work.
