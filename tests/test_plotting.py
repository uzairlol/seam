"""Unit tests for plotting functions in seam.analysis.plotting."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import pandas as pd

from seam.analysis.plotting import (
    plot_collapse_performance_tradeoff,
    plot_contamination_propagation,
    plot_metric_condition_heatmap,
    plot_memory_collapse,
    plot_performance_comparison,
    plot_poisoning_performance_comparison,
    plot_run_level_distributions,
)


def _load_generate_figures():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "generate_figures.py"
    spec = importlib.util.spec_from_file_location("generate_figures_module", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_figures


def test_plotting_routines():
    df_sample = pd.DataFrame([
        {
            "policy": "naive_overwrite",
            "topology": "full_broadcast",
            "poisoning_mode": "clean",
            "final_score": 0.85,
            "mean_self_bleu": 0.20,
            "peer_contamination_rate": 0.0,
        },
        {
            "policy": "structured_incremental",
            "topology": "ring",
            "poisoning_mode": "internal",
            "final_score": 0.45,
            "mean_self_bleu": 0.65,
            "peer_contamination_rate": 0.75,
        },
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        p1 = plot_performance_comparison(df_sample, tmp_path / "perf.png")
        p2 = plot_memory_collapse(df_sample, tmp_path / "collapse.png")
        p3 = plot_contamination_propagation(df_sample, tmp_path / "contam.png")
        p4 = plot_metric_condition_heatmap(df_sample, tmp_path / "heatmap.png")
        p5 = plot_poisoning_performance_comparison(df_sample, tmp_path / "poison.png")
        p6 = plot_collapse_performance_tradeoff(df_sample, tmp_path / "tradeoff.png")
        p7 = plot_run_level_distributions(df_sample, tmp_path / "distributions.png")

        assert all(path.exists() for path in (p1, p2, p3, p4, p5, p6, p7))


def test_new_plotting_routines_handle_missing_columns() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        incomplete = pd.DataFrame({"policy": ["naive_overwrite"]})

        outputs = [
            plot_metric_condition_heatmap(incomplete, tmp_path / "heatmap.png"),
            plot_poisoning_performance_comparison(incomplete, tmp_path / "poison.png"),
            plot_collapse_performance_tradeoff(incomplete, tmp_path / "tradeoff.png"),
            plot_run_level_distributions(incomplete, tmp_path / "distributions.png"),
        ]

        assert all(path.exists() for path in outputs)


def test_generate_figures_exports_summary_statistics():
    df_sample = pd.DataFrame([
        {
            "run_id": "r1",
            "policy": "naive_overwrite",
            "topology": "ring",
            "poisoning_mode": "clean",
            "seed": 42,
            "final_score": 0.85,
            "mean_self_bleu": 0.20,
            "peer_contamination_rate": 0.0,
        },
        {
            "run_id": "r2",
            "policy": "naive_overwrite",
            "topology": "ring",
            "poisoning_mode": "clean",
            "seed": 43,
            "final_score": 0.75,
            "mean_self_bleu": 0.30,
            "peer_contamination_rate": 0.1,
        },
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        df_sample.to_csv(runs_dir / "results_summary.csv", index=False)

        figures_dir = tmp_path / "figures"
        generate_figures = _load_generate_figures()
        generate_figures(input_dir=str(runs_dir), figures_dir=str(figures_dir))

        stats_csv = figures_dir / "summary_statistics.csv"
        table_md = figures_dir / "summary_table.md"

        assert stats_csv.exists()
        assert table_md.exists()

        stats_df = pd.read_csv(stats_csv)
        assert "final_score_ci_low" in stats_df.columns
        assert "final_score_ci_high" in stats_df.columns
