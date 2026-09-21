"""Unit tests for run_experiments.py in scripts/run_experiments.py."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import pandas as pd


def _load_run_experiments_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_experiments.py"
    spec = importlib.util.spec_from_file_location("run_experiments_module", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_run_experiments = _load_run_experiments_module()
_save_summary_manifest = _run_experiments._save_summary_manifest
generate_experiment_grid = _run_experiments.generate_experiment_grid
run_experiments = _run_experiments.run_experiments


def test_generate_experiment_grid():
    policies = ["naive_overwrite", "structured_incremental"]
    sharing = ["off", "ring"]
    poisoning = ["clean", "internal"]
    seeds = [42, 43]

    grid = generate_experiment_grid(
        policies=policies,
        sharing_modes=sharing,
        poisoning_modes=poisoning,
        seeds=seeds,
    )

    # 2 policies * 2 sharing * 2 poisoning * 2 seeds = 16 conditions
    assert len(grid) == 16
    assert grid[0]["policy"] == "naive_overwrite"
    assert grid[0]["seed"] == 42


def test_save_summary_manifest():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir)
        sample_results = [
            {
                "run_id": "run_1",
                "experiment_id": "exp_1",
                "policy": "naive_overwrite",
                "topology": "ring",
                "poisoning_mode": "clean",
                "seed": 42,
                "rounds_played": 10,
                "final_score": 0.85,
                "mean_self_bleu": 0.12,
                "peer_contamination_rate": 0.0,
            }
        ]

        _save_summary_manifest(sample_results, out_path)

        csv_file = out_path / "results_summary.csv"
        manifest_file = out_path / "manifest.json"

        assert csv_file.exists()
        assert manifest_file.exists()

        df = pd.read_csv(csv_file)
        assert len(df) == 1
        assert df.iloc[0]["experiment_id"] == "exp_1"
        assert df.iloc[0]["final_score"] == 0.85


def test_run_experiments_with_sampling_trials(tmp_path: Path, monkeypatch):
    """trials>1 emits one run per (condition, seed) with distinct decoding seeds."""

    class _FakeRunner:
        def __init__(self, config, seed=None, base_dir="runs"):
            self.config = config
            self.seed = seed

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def run(self):
            return {
                "run_id": f"mock_{self.config.experiment_id}_{self.seed}",
                "seed": self.seed,
                "rounds_played": 2,
                "final_score": 0.3,
                "oracle_score": 0.8,
                "mean_self_bleu": 0.5,
            }

    monkeypatch.setattr(_run_experiments, "EpisodeRunner", _FakeRunner)

    results = run_experiments(
        env_type="resource_foraging",
        policies=["no_memory"],
        sharing_modes=["off"],
        poisoning_modes=["clean"],
        seeds=[42],
        output_dir=str(tmp_path),
        trials=2,
        sample_temperature=0.6,
    )

    assert len(results) == 2
    trials_seen = sorted(r["trial"] for r in results)
    assert trials_seen == [0, 1]
    assert len({r["run_id"] for r in results}) == 2
    exp_ids = sorted(r["experiment_id"] for r in results)
    assert exp_ids[0].endswith("_t0") and exp_ids[1].endswith("_t1")

    csv = tmp_path / "results_summary.csv"
    df = pd.read_csv(csv)
    assert len(df) == 2
    assert "trial" in df.columns
    assert "oracle_score" in df.columns
