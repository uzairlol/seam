# Audit History and Result Validity

## Pre-fix issues identified

The earlier audit identified risks in channel poisoning, Self-BLEU context mixing, substring contamination matching, bargaining score naming, sharing cadence, memory limits, deprecated-rule retention, poison reward scale, experiment-grid semantics, logging, action extraction, and artifact truncation.

The current code includes fixes for these areas, including peer-targeted channel injection, local-context Self-BLEU input, boundary-aware phrase detection, round-1 sharing cadence, bounded memory text, deprecated-rule cleanup, normalized poison reward, prompt retention, and safer action matching.

## How to classify runs

### Pre-fix

Runs generated before the audit commit should be treated as historical/debugging artifacts. They may be useful for demonstrating why the fixes were needed, but should not be pooled into final evidence.

### Post-fix

A run is eligible for the main result set only when its config snapshot corresponds to the fixed code, it has a complete summary, and its condition/seed identity is verified.

## Sanity checks before analysis

- Clean conditions have no peer contamination.
- Sharing-off conditions have no peer contamination, even under internal poisoning.
- Every expected policy/topology/poisoning/seed combination exists exactly once.
- `manifest.json`, `results_summary.csv`, and run directories agree.
- `events.jsonl` has a coherent number of agent-round events.
- The saved config matches the intended model, environment, episode length, and sharing cadence.
- Generated figures were produced from the intended experiment directory rather than a stale parent CSV.

## Known interpretation limits

A 100% contamination rate means every peer had at least one detected phrase somewhere in its memory sequence. It does not mean every peer was contaminated at the same round or that the poison caused every observed performance change. Likewise, a high Self-BLEU score is evidence of lexical similarity, not a complete diagnosis of memory failure.

## Reporting recommendation

Keep an audit table beside the manuscript data that lists commit, run date, model tag, seed set, condition count, missing runs, and analysis command. This makes it possible to distinguish a scientific result from an artifact of an older implementation.

## Practical audit record

For each result release, preserve:

| Field | Example meaning |
|---|---|
| Source commit | Code version that produced the runs |
| Model tag | Exact Ollama model identifier |
| Environments | Scenario directories included |
| Conditions | Policy/topology/poisoning combinations |
| Seeds | Exact seed values, not only count |
| Completed runs | Number of valid `summary.json` files |
| Exclusions | Missing, interrupted, or invalid folders |
| Analysis command | Exact figure/table generation command |

This record should be generated or maintained before manuscript tables are copied into prose. It is especially important because an experiment directory can contain valid run folders while its cached `results_summary.csv` or `manifest.json` still describes an earlier subset.

## Distinguishing a code error from a result pattern

A surprising result is not automatically a bug. First check the saved configuration and event sequence, then compare against a clean control and sharing-off control. A likely implementation error is indicated when an invariant is violated, such as contamination in a clean run, peer contamination with sharing disabled, a missing condition row, or an incomplete event count. A legitimate result may instead show high variance, early episode termination, or a policy-specific response pattern while preserving those invariants.
