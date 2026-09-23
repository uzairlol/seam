# Outputs and Reproducibility

## Prerequisites

- Python 3.11 or newer.
- Dependencies from `environment.yml`, `requirements.txt`, or the project metadata.
- Ollama running locally at the configured base URL.
- The requested model pulled into Ollama, for example `qwen2.5:7b` or the exact tag configured in the run.

Use the environment selected for the project. Do not mix Python environments between runs and analysis.

## Baselines

```powershell
python scripts/run_baselines.py --model qwen2.5:7b --env resource_foraging --seeds 40 41 42 43 44 45 46 47 48 49 --outdir runs/baselines/resource_foraging
python scripts/run_baselines.py --model qwen2.5:7b --env bargaining_game --seeds 40 41 42 43 44 45 46 47 48 49 --outdir runs/baselines/bargaining_game
python scripts/run_baselines.py --model qwen2.5:7b --env number_guessing --seeds 40 41 42 43 44 45 46 47 48 49 --outdir runs/baselines/number_guessing
```

Note on seeds: the shipped baseline report (`reports/baselines/`) and the statistical suite both assume seeds 40-49, so keep the seed list identical across the baseline tier and the factorial tier. Because the efficacy gap is normalized per seed against the mean `no_memory` score of that same seed, a factorial seed that has no baseline row yields a NaN efficacy gap. The `--outdir` argument is placement advice; the baseline script itself records `env_type` per run in `metadata.json`, so the flat `runs/baselines/` layout from the delivered 120-run tier is equally consumable by `analyze_baselines.py`, which keys on that field rather than the directory path.

## Factorial experiments

```powershell
python scripts/run_experiments.py --model qwen2.5:7b --env resource_foraging --policies naive_overwrite raw_trajectory_buffer structured_incremental --sharing off full_broadcast ring --poisoning clean internal --seeds 40 41 42 43 44 45 46 47 48 49 --outdir runs/experiments/resource_foraging
```

Repeat the same command with `--env bargaining_game` and `--env number_guessing` and their corresponding output directories. Be aware that the CLI defaults are wider than this canonical command (they include `no_memory`, the `star`/`cluster` topologies, the `channel`/`gradual` poisoning modes, and seeds 42-51); when the paper reports a grid, quote the exact flag values actually used rather than the argument defaults.

## Figure generation

```powershell
python scripts/generate_figures.py --indir runs/experiments/resource_foraging --outdir figures/resource_foraging
python scripts/generate_figures.py --indir runs/experiments/bargaining_game --outdir figures/bargaining_game
python scripts/generate_figures.py --indir runs/experiments/number_guessing --outdir figures/number_guessing
```

`generate_figures.py` auto-detects `runs/baselines/<env>` as the efficacy-gap control when it exists; pass `--baseline-dir` to override.

## Other analysis scripts

Four additional scripts consume the same run artifacts and should appear in, or at least be consistent with, the reproducibility appendix.

`scripts/analyze_baselines.py --indir runs/baselines --figdir figures/baselines --outdir reports/baselines` rehydrates baseline runs, computes per-environment efficacy gaps, runs policy-versus-control statistics, and writes `figures/baselines/<env>/*.png`, `figures/baselines/cross_environment_*.png`, `reports/baselines/summary_statistics.csv`, `reports/baselines/summary_table.md`, and `reports/baselines/statistical_tests.csv`. It reproduces the shipped baseline report and its tables from raw runs.

`python scripts/run_significance_tests.py` runs the invariant audit plus the H1-H7 Mann-Whitney suite across every environment directory found under `runs/experiments/<env>` and writes `reports/audit_invariants.json`, `reports/statistical_tests.{json,csv,md}`. It does not merge the sibling baseline directory itself, so plan the experiment grid to include `no_memory` rows (or pass a pre-merged DataFrame) for complete efficacy-gap analyses.

`python scripts/analyze_bargaining_metrics.py` reads bargaining event logs and writes `reports/bargaining_detailed_metrics.csv` and `reports/bargaining_metric_clarification.md`, adding acceptance rate and total-welfare columns that the fairness score alone cannot express.

`python scripts/extract_case_studies.py` discovers foraging runs by folder-name pattern and writes `reports/qualitative_traces.{json,md}` containing per-round memory-state excerpts for manuscript case studies; it never hardcodes a specific run directory.

All of these scripts read only committed artifact formats (metadata, config snapshot, events, summary) and are safe to re-run after any completed sweep; none of them writes back into the run directories.

## Run-directory contract

Each completed run should contain:

```text
<run_id>/
  metadata.json
  config_snapshot.yaml
  events.jsonl
  summary.json
```

An experiment directory may also contain:

```text
results_summary.csv
manifest.json
```

The experiment-level CSV and manifest should contain the same completed-run count. If folders, CSV, and manifest disagree, regenerate or audit before reporting.

## Reproducibility checklist

- Record commit hash and model tag.
- Record all seeds and expected condition count.
- Verify every run has `summary.json`.
- Verify every condition has exactly one row per seed.
- Check `rounds_played` for early termination versus maximum length.
- Regenerate summaries and figures only after all intended runs finish.
- Preserve raw run directories alongside aggregate tables.
- Separate pre-fix and post-fix results; never pool them.
- Run the test suite before changing the interpretation of a result.

## Useful inspection pattern

For a single run, inspect files in this order:

1. `metadata.json` for run ID, environment, seed, model, policy, topology, and poisoning mode.
2. `config_snapshot.yaml` for the complete Pydantic configuration captured at initialization.
3. `summary.json` for final metrics and completion timestamp.
4. `events.jsonl` for per-agent, per-round behavior and prompt/memory evidence.

The rehydrator can load metadata, configuration, events, and summary independently and can convert events into a pandas DataFrame. Missing metadata/configuration is an integrity failure; missing events can be valid only for a deliberately minimal artifact, not for a normal completed episode.

## Version and model controls

The repository provides helpers for obtaining a short Git commit hash and `pip freeze`, but callers must explicitly record or inspect those values. A manuscript-quality run record should also preserve the exact Ollama model tag, base URL, decoding parameters, environment settings, seed list, and source commit. Changing any of these can change results even when the nominal experiment condition is unchanged.
