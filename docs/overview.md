# Project Overview

## Purpose

SEAM (Shared Evolving Agent Memory) is a controlled study of self-evolving LLM agents that update memory after interaction and optionally exchange memory artifacts with peers. The project asks whether communication improves learning, accelerates memory collapse, or spreads harmful but plausible experience.

## Central question

> When independently evolving agents share memory, does exchange stabilize behavior, create repetition, or propagate contamination through the population?

## Experimental factors

The implemented experiment grid crosses:

- Memory policy: `naive_overwrite`, `raw_trajectory_buffer`, `structured_incremental`
- Sharing topology: `off`, `full_broadcast`, `ring` (the CLI accepts topology-like values and maps them to sharing mode/topology pairs)
- Poisoning condition: `clean`, `internal`
- Environment: `resource_foraging`, `bargaining_game`, `number_guessing`
- Seed: repeated independent episode runs

The current experiment script constructs 4-agent, 20-round conditions unless a caller changes the script/API. The environment YAML files are useful configuration references, but the CLI script values govern script-launched experiments.

## Research concepts

### Memory collapse

A memory policy repeatedly rewrites or accumulates experience. Collapse is operationalized primarily through high temporal Self-BLEU, with action entropy and memory length available as complementary behavioral and storage signals.

### Contamination

A poisoned lesson begins in one seed agent and may enter peer memory through sharing. The implemented peer contamination rate asks whether any peer memory state contains a configured poison phrase during the episode.

### Performance

Each environment computes an objective score from its own state. No LLM judge is used by the environments.

## Repository map

- `src/seam/agents/`: prompt construction, model calls, action extraction, and population management.
- `src/seam/envs/`: deterministic task environments.
- `src/seam/memory/`: memory-policy implementations and factory.
- `src/seam/sharing/`: communication engine and topology generation.
- `src/seam/poisoning/`: poison payload selection and injection.
- `src/seam/orchestration/`: Pydantic configuration and episode integration.
- `src/seam/metrics/`: collapse and contamination calculations.
- `src/seam/logging/`: JSONL logging and run rehydration.
- `src/seam/analysis/`: run aggregation and plotting.
- `scripts/`: baseline, factorial experiment, and figure-generation entry points.
- `configs/`: environment and model YAML references.
- `runs/`: generated run artifacts; these should be treated as data, not source.
- `figures/`: generated plots and aggregate tables.
- `literature/`: converted papers and project-scope documents.

## Claim discipline

A result should identify its environment, policy, topology, poisoning mode, model, episode length, seed set, and whether it was generated before or after a code fix. A high-level project claim is not evidence until it can be traced to run-level artifacts and the current aggregation procedure.

## Research unit

The basic unit of analysis is one **episode run**: one environment, one model configuration, one memory policy per agent, one sharing configuration, one poisoning configuration, and one seed. A condition is a group of runs that share the non-seed factors. The aggregate table summarizes conditions; it does not replace the run-level observations.

## What the project does not claim by default

SEAM does not automatically establish that sharing causes a performance change, that high Self-BLEU proves cognitive collapse, or that phrase presence proves semantic adoption of poison. Those claims require additional controls or analyses. The current design supports controlled descriptive comparisons and targeted sanity checks; causal and semantic claims need to be stated with care.
