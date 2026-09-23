# Artifact Schemas

This chapter is the reproducibility-appendix knowledge dump: every file that a run sweep writes, field by field, with the exact semantics of each value. Four artifact tiers exist: per-run files inside one run directory, sweep-level files at the experiment directory root, analysis output files under `reports/`, and figure files under `figures/`. The `RunLogger` writes the first tier; the CLIs write the second; the analysis scripts write the third and fourth.

## Per-run directory

A single run writes exactly four files into its own directory (the directory name equals the run ID): `metadata.json`, `config_snapshot.yaml`, `events.jsonl`, and `summary.json`. `events.jsonl` is opened in append mode for every logged step; the other three are written once (metadata and config at run start, summary at the moment the episode ends), so a run that crashes mid-episode will have metadata and config snapshot but no `summary.json`, and its events will be a prefix of the intended stream. This asymmetry is exactly what the aggregator relies on to detect incomplete runs: every analysis script keys on the existence of `summary.json`, never on the events file.

### Run ID and folder naming

`RunLogger` composes the run ID as `<experiment_id>_seed<seed>_<UTC timestamp YYYYMMDD_HHMMSS>`, and the run directory is `<base_dir>/<run_id>`. The experiment ID comes from the config: the factorial CLI uses `exp_<policy>_<topology>_<poisoning_mode>` (with `_t<trial>` appended when `--trials > 1`), and the baseline CLI uses `baseline_<policy>`. Because the timestamp has one-second resolution, two runs of the same condition started within the same second could collide folder names even though `run_id` in memory is unique per process; in practice each condition runs exactly once so collisions have not occurred, but the bookkeeping that detects "the same condition ran twice" should compare `metadata.json` fields, not folder names.

### metadata.json

| Field | Type | Content |
|---|---|---|
| `run_id` | str | The run identifier (also the folder naarrayame). |
| `experiment_id` | str | `exp_*` or `baseline_*` condition identifier. |
| `seed` | int | The episode seed (also the env seed). |
| `timestamp` | str | ISO-8601 UTC start timestamp. |
| `git_commit` | str | Short commit hash from `get_git_commit_hash()`. |
| `env_type` | str | `resource_foraging`, `bargaining_game`, or `number_guessing`. |
| `n_agents` | int | Agent count from `EnvConfig` (6 in all shipped runs). |
| `model_name` | str | Ollama model tag (e.g. `qwen2.5:7b`). |
| `memory_policy` | str | Policy name, e.g. `structured_incremental`. |
| `sharing_mode` | str | `off`, `broadcast`, `selective`, or `peer_to_peer`. |
| `topology` | str | The configured topology; **defaults to `full_broadcast` even when sharing is `off`** (see the lifecycle chapter). |
| `poisoning_mode` | str | `clean`, `internal`, `channel`, or `gradual`. |

The `git_commit` field is the single strongest provenance guard in the whole artifact set: it lets a later reader detect whether a run was produced by the fixed code, which is exactly what the audit-history chapter requires for the pre-fix/post-fix split.

### config_snapshot.yaml

This is `ExperimentConfig.model_dump()` serialized as YAML: the full `env` block (including the unused-by-default `rich_cell_yield`), the complete `model` block, the `memory` block, the `sharing` block (mode, topology, `publish_every_n_rounds`, `max_artifact_tokens`, `consume_mode`), the `poisoning` block, and the `seeds` list. It is the ground truth for "what was configured", while `metadata.json` is a flattened convenience projection of the same object; any conflict between the two should be resolved in favour of the snapshot. Note that in the shipped factorial runs `publish_every_n_rounds` is 2 (set by the CLI) while the `SharingConfig` schema default is 3, and `consume_mode` stays at the default `all`.

### events.jsonl

One JSON object per agent per round, written in execution order (round-outer, agent-inner), each on its own line. The `round` field is 1-indexed, and the expected total line count equals `rounds_played × n_agents` when every agent logs every round — which holds in all three environments, including bargaining.

| Field | Type | Content |
|---|---|---|
| `round` | int | 1-indexed round number. |
| `agent_id` | str | Agent identifier. |
| `observation` | dict | The observation dict exactly as handed to the agent. |
| `prompt` | str | The exact prompt string sent to the model (authoritative). |
| `raw_response` | str | **The parsed action, not the raw model text** (known asymmetry, see the prompts chapter). |
| `action` | str | The parsed, validated action. |
| `reward` | float | Scalar reward granted this step. |
| `memory_state` | str | The updated memory context after this round's update. |
| `latency_ms` | int | Always `0` in current runs (the runner never passes a real latency). |
| `info` | dict | Extra step diagnostics from the environment (often empty). |
| `timestamp` | str | ISO-8601 UTC event time. |

### summary.json

The summary is a flat envelope with `run_id`, `final_score`, `summary_info`, and `completed_at`. All the interesting numbers live inside `summary_info`, which is the dict returned by the runner:

| Field | Type | Content |
|---|---|---|
| `run_id` | str | Run identifier. |
| `seed` | int | Episode seed. |
| `rounds_played` | int | Actual rounds executed (number guessing often ends early). |
| `final_score` | float | `get_ground_truth_score()` of the episode. |
| `oracle_score` | float | Deterministic oracle score on the same `(env_type, seed)` instance. |
| `cumulative_rewards` | dict[str, float] | Per-agent sum of rewards. |
| `mean_self_bleu` | float | Mean over agents of intra-agent Self-BLEU on **local** memory. |
| `per_agent_self_bleu` | dict[str, float] | Per-agent Self-BLEU. |
| `per_agent_action_entropy` | dict[str, list[float]] | Per-agent rolling action-entropy windows. |
| `per_agent_memory_lengths` | dict[str, list[int]] | Per-agent word-count series over rounds. |
| `peer_contamination_rate` | float | Fraction of non-seed agents ever contaminated; 0.0 unless poisoning active and sharing on. |
| `per_agent_poison_adherence` | dict[str, float] | Per-agent fraction of actions matching the payload. |
| `per_agent_poison_dosage` | dict[str, float] | Per-agent peak fractional memory dosage. |
| `poison_dosage_rate` | float | Mean over peers dosage; 0.0 unless poisoning active and sharing on. |
| `propagation_latency` | float \| null | Mean first-contamination round over peers; null if no peer ever contaminated. |
| `peer_propagation_round` | dict[str, int] | First contaminated round per peer (or its absence). |

Three values need the caveats spelled out for the paper. `mean_self_bleu` is computed on the local memory context recorded *before* each round's update, so shared peer context is intentionally excluded from the collapse measure. `final_score` is the environment truth score, not a normalized value; comparability across environments comes from `efficacy_gap`, which is not stored in the summary at all and is derived in the analysis tier. `propagation_latency` averages only the peers that did contaminate; peers that stayed clean are not counted in the mean but their identity is visible in `peer_propagation_round`.

## Sweep-level files

The factorial and baseline CLIs write two aggregated files into their output directory: `results_summary.csv` and `manifest.json`. Both are derived post-hoc from the in-memory run summaries, not from the run directories, and therefore both are stale the moment an additional run is added to the same directory without re-running the CLI. The aggregator's directory scan treats them with suspicion for exactly this reason: when the scan of `summary.json` files finds more rows than the CSV, the CSV is ignored and rehydration wins.

`results_summary.csv` has one row per run with the columns `run_id`, `experiment_id`, `policy`, `topology`, `poisoning_mode`, `seed`, `trial` (0 for single-trial runs), `rounds_played`, `final_score`, `oracle_score`, `mean_self_bleu`, `peer_contamination_rate`, `poison_dosage_rate`, `propagation_latency`. `manifest.json` stores the same information as a JSON list under a `results` key with a `total_runs` count; it is the JSON twin of the CSV, and the audit-history chapter lists "manifest, CSV and run directories agree" as a required sanity check.

## Analysis output files

All readers below consume `metadata.json` + `summary.json` through `RunRehydrator` or the aggregator. `reports/baselines/summary_statistics.csv` and `reports/baselines/summary_table.md` come from `analyze_baselines.py`: the CSV has per-environment-per-policy rows with `n_runs` and, for `final_score`, `efficacy_gap`, `mean_self_bleu` and `oracle_score`, the mean, sample std, SEM and 95% t-CI (clipped to `[0,1]` for the bounded metrics, unclipped for `efficacy_gap`), plus `peer_contamination_rate_mean` and `poison_dosage_rate_mean`. The Markdown table is the human-readable projection with score/oracle/gap/BLEU columns. `reports/baselines/statistical_tests.csv` holds the baseline-tier Mann-Whitney tests with FDR-corrected p-values (see the statistical-analysis chapter for its semantics); the experiment-tier equivalent is produced by `run_significance_tests.py`, which writes `reports/statistical_tests.{json,csv,md}` and `reports/audit_invariants.json`. The bargaining-specific script contributes `reports/bargaining_detailed_metrics.csv` and `reports/bargaining_metric_clarification.md`, and the case-study script contributes `reports/qualitative_traces.{json,md}`.

## Figure files

`generate_figures.py` and `analyze_baselines.py` write only PNG files (300 dpi, tight bounding box) and never touch the run directories. Factorial figures land under `figures/<env>/` named for the seven shipped plots; baseline figures land under `figures/baselines/<env>/` and `figures/baselines/cross_environment_*.png`. Because figures are regenerated from the same analysis functions, re-running either script on a corrected directory reproduces the identical charts except for matplotlib version drift.