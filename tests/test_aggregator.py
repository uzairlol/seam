"""Unit tests for ResultAggregator in seam.analysis.aggregator."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from seam.analysis.aggregator import ResultAggregator


def _write_run_dir(
    run_dir: Path,
    run_id: str,
    *,
    policy: str,
    topology: str,
    poisoning_mode: str,
    seed: int,
    final_score: float,
    mean_self_bleu: float,
    contamination_rate: float,
    propagation_latency: float | None,
) -> None:
    """Create a minimal run directory that RunRehydrator can load."""
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "run_id": run_id,
        "experiment_id": "exp_test",
        "seed": seed,
        "env_type": "resource_foraging",
        "n_agents": 6,
        "model_name": "qwen2.5:7b",
        "memory_policy": policy,
        "sharing_mode": "broadcast",
        "topology": topology,
        "poisoning_mode": poisoning_mode,
    }
    summary = {
        "run_id": run_id,
        "final_score": final_score,
        "summary_info": {
            "mean_self_bleu": mean_self_bleu,
            "peer_contamination_rate": contamination_rate,
            "propagation_latency": propagation_latency,
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_aggregator_with_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        df_sample = pd.DataFrame(
            [
                {
                    "run_id": "r1",
                    "policy": "naive_overwrite",
                    "topology": "ring",
                    "poisoning_mode": "clean",
                    "seed": 42,
                    "final_score": 0.8,
                    "mean_self_bleu": 0.1,
                    "peer_contamination_rate": 0.0,
                },
                {
                    "run_id": "r2",
                    "policy": "naive_overwrite",
                    "topology": "ring",
                    "poisoning_mode": "clean",
                    "seed": 43,
                    "final_score": 0.9,
                    "mean_self_bleu": 0.2,
                    "peer_contamination_rate": 0.0,
                },
            ]
        )
        df_sample.to_csv(tmp_path / "results_summary.csv", index=False)

        agg = ResultAggregator(tmp_path)
        summary_df = agg.aggregate_conditions()

        assert len(summary_df) == 1
        assert pytest.approx(summary_df.iloc[0]["final_score_mean"]) == 0.85
        assert "final_score_sem" in summary_df.columns
        assert "final_score_ci_low" in summary_df.columns
        assert "final_score_ci_high" in summary_df.columns
        assert summary_df.iloc[0]["final_score_ci_low"] <= summary_df.iloc[0]["final_score_mean"]
        assert summary_df.iloc[0]["final_score_ci_high"] >= summary_df.iloc[0]["final_score_mean"]

        md_table = agg.to_markdown_table()
        assert "| naive_overwrite | ring | clean |" in md_table
        assert "95% CI" in md_table


def test_aggregator_rehydrates_runs_with_propagation_latency():
    """The directory scan extracts propagation_latency so latency hypotheses can run."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        run_dir = tmp_path / "exp_run_seed42_20260921"
        _write_run_dir(
            run_dir,
            "exp_run_seed42_20260921",
            policy="structured_incremental",
            topology="ring",
            poisoning_mode="internal",
            seed=42,
            final_score=0.12,
            mean_self_bleu=0.99,
            contamination_rate=1.0,
            propagation_latency=3.0,
        )

        agg = ResultAggregator(tmp_path)
        assert len(agg.df) == 1
        row = agg.df.iloc[0]
        assert row["policy"] == "structured_incremental"
        assert row["propagation_latency"] == 3.0


def test_aggregator_prefers_scan_when_csv_is_stale():
    """A stale results_summary.csv (fewer rows than run directories) must be ignored."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Three completed run directories...
        for seed in (42, 43, 44):
            _write_run_dir(
                tmp_path / f"run_{seed}",
                f"run_{seed}",
                policy="naive_overwrite",
                topology="ring",
                poisoning_mode="clean",
                seed=seed,
                final_score=0.5,
                mean_self_bleu=0.95,
                contamination_rate=0.0,
                propagation_latency=None,
            )
        # ...but a CSV that only describes a single (older) run.
        pd.DataFrame(
            [
                {
                    "run_id": "old_run",
                    "policy": "no_memory",
                    "topology": "off",
                    "poisoning_mode": "clean",
                    "seed": 42,
                    "final_score": 0.0,
                    "mean_self_bleu": 0.0,
                    "peer_contamination_rate": 0.0,
                }
            ]
        ).to_csv(tmp_path / "results_summary.csv", index=False)

        agg = ResultAggregator(tmp_path)
        assert len(agg.df) == 3
        assert set(agg.df["policy"].unique()) == {"naive_overwrite"}


def test_aggregator_uses_csv_when_complete():
    """A CSV that matches the completed run-directory count is kept as ground truth."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for seed in (42, 43):
            _write_run_dir(
                tmp_path / f"run_{seed}",
                f"run_{seed}",
                policy="naive_overwrite",
                topology="ring",
                poisoning_mode="clean",
                seed=seed,
                final_score=0.5,
                mean_self_bleu=0.95,
                contamination_rate=0.0,
                propagation_latency=None,
            )
        pd.DataFrame(
            [
                {
                    "run_id": f"run_{seed}",
                    "policy": "naive_overwrite",
                    "topology": "ring",
                    "poisoning_mode": "clean",
                    "seed": seed,
                    "final_score": 0.5,
                    "mean_self_bleu": 0.95,
                    "peer_contamination_rate": 0.0,
                }
                for seed in (42, 43)
            ]
        ).to_csv(tmp_path / "results_summary.csv", index=False)

        agg = ResultAggregator(tmp_path)
        assert len(agg.df) == 2
        assert "run_42" in set(agg.df["run_id"])


def test_aggregator_derives_efficacy_gap_from_oracle_scores():
    """Scan-based data with oracle_score + a no_memory baseline yields efficacy_gap."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for run_id, policy, topo, score in (
            ("ctl_42", "no_memory", "off", 0.1),
            ("treated_42", "naive_overwrite", "ring", 0.3),
        ):
            run_dir = tmp_path / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "experiment_id": "exp_t",
                        "seed": 42,
                        "memory_policy": policy,
                        "sharing_mode": "broadcast",
                        "topology": topo,
                        "poisoning_mode": "internal",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "final_score": score,
                        "summary_info": {
                            "mean_self_bleu": 0.9,
                            "peer_contamination_rate": 0.5,
                            "propagation_latency": 2.0,
                            "oracle_score": 0.5,
                            "poison_dosage_rate": 0.3,
                        },
                    }
                ),
                encoding="utf-8",
            )

        agg = ResultAggregator(tmp_path)
        assert "oracle_score" in agg.df.columns
        assert "poison_dosage_rate" in agg.df.columns
        assert "efficacy_gap" in agg.df.columns
        # efficacy = (0.3 - 0.1) / (0.5 - 0.1) = 0.5 for the treated run
        treated = agg.df[agg.df["run_id"] == "treated_42"]
        assert pytest.approx(treated.iloc[0]["efficacy_gap"]) == 0.5

        summary_df = agg.aggregate_conditions()
        assert "efficacy_gap_mean" in summary_df.columns
        md = agg.to_markdown_table()
        assert "Efficacy Gap" in md
        assert "Oracle Score" in md
