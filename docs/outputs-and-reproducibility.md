# Outputs and Reproducibility

## Prerequisites

- Python 3.11 or newer.
- Dependencies from `environment.yml`, `requirements.txt`, or the project metadata.
- Ollama running locally at the configured base URL.
- The requested model pulled into Ollama, for example `qwen2.5:7b` or the exact tag configured in the run.

Use the environment selected for the project. Do not mix Python environments between runs and analysis.

## Baselines

```powershell
python scripts/run_baselines.py --model qwen2.5:7b --env resource_foraging --seeds 45 46 47 48 49 --outdir runs/baselines/resource_foraging
python scripts/run_baselines.py --model qwen2.5:7b --env bargaining_game --seeds 45 46 47 48 49 --outdir runs/baselines/bargaining_game
python scripts/run_baselines.py --model qwen2.5:7b --env number_guessing --seeds 45 46 47 48 49 --outdir runs/baselines/number_guessing
```

## Factorial experiments

```powershell
python scripts/run_experiments.py --model qwen2.5:7b --env resource_foraging --policies naive_overwrite raw_trajectory_buffer structured_incremental --sharing off full_broadcast ring --poisoning clean internal --seeds 45 46 47 48 49 --outdir runs/experiments/resource_foraging
```

Repeat the same command with `--env bargaining_game` and `--env number_guessing` and their corresponding output directories.

## Figure generation

```powershell
python scripts/generate_figures.py --indir runs/experiments/resource_foraging --outdir figures/resource_foraging
python scripts/generate_figures.py --indir runs/experiments/bargaining_game --outdir figures/bargaining_game
python scripts/generate_figures.py --indir runs/experiments/number_guessing --outdir figures/number_guessing
```

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
