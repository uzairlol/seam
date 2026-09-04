# Experiment Lifecycle

## Configuration

Pydantic models in `src/seam/orchestration/config_loader.py` define model, environment, memory, sharing, poisoning, and experiment settings. YAML files can be loaded with `load_experiment_config()`. The CLI scripts also construct configurations directly.

## Baseline workflow

`scripts/run_baselines.py` loops over the three policies, constructs a clean no-sharing configuration, and runs each requested seed through `EpisodeRunner`. Baselines are single-agent in conceptual purpose, although the current script configuration creates four agents with sharing disabled; document this exact setting when reporting results.

## Factorial workflow

`scripts/run_experiments.py` creates one condition for each policy x sharing topology x poisoning mode x seed combination. The standard requested matrix is 3 x 3 x 2 x N. Each condition gets its own run directory and summary.

## Per-round loop

```mermaid
sequenceDiagram
    participant R as EpisodeRunner
    participant S as SharingEngine
    participant P as Population
    participant E as Environment
    participant M as MemoryPolicy
    participant L as RunLogger

    R->>S: inject/reroute shared artifacts
    R->>P: act(observation, local + shared context)
    P->>R: parsed actions
    R->>E: step(actions)
    E->>R: observations, rewards, done, info
    R->>M: update(experience, shared context)
    R->>L: log prompt, action, reward, memory
```

## End-of-run evaluation

The runner computes final objective score, cumulative rewards, Self-BLEU, action entropy, memory lengths, peer contamination, and poison adherence. It writes these into the summary structure and records the episode end through `RunLogger`.

## Figure workflow

`scripts/generate_figures.py` loads `results_summary.csv` when present, otherwise scans run folders for `summary.json`. It groups data by policy, topology, and poisoning mode and writes three PNGs plus a CSV and Markdown table.

Because CSV takes precedence, regenerate the experiment-level summary before plotting after adding new run folders. Otherwise, a stale CSV can omit valid later runs.

## Seed accounting

For each environment, the number of expected experiment runs is:

`number of policies x number of topologies x number of poisoning modes x number of seeds`.

For the standard 3 x 3 x 2 grid and seeds 42-49, this is 144 runs. For only seeds 45-49, it is 90 runs. Baseline expectations are 3 policies x number of seeds.

## Failure and completion behavior

The run logger creates the run directory early, then appends one JSON object per agent step to `events.jsonl`. The final `summary.json` is written only when `log_episode_end()` is reached. A directory with metadata and partial events but no summary should be treated as incomplete, even if its name looks like a completed run.

## Counting events

For a run with `A` agents and `R` completed rounds, the expected event count is normally `A * R`. Early success can make `R` smaller than the configured maximum. This count is a useful low-cost integrity check, but it does not prove that prompts, actions, observations, and memory updates were semantically correct.

## Analysis precedence

The aggregator's CSV-first behavior is deliberate for speed but creates a reproducibility hazard when a CSV predates additional run folders. A robust workflow is to finish all runs, verify folder counts and summaries, then regenerate the experiment-level CSV/manifest and figures in a separate analysis step.
