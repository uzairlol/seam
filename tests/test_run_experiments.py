"""Unit tests for run_experiments.py in scripts/run_experiments.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
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
