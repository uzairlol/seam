"""Unit tests for plotting functions in seam.analysis.plotting."""

from __future__ import annotations

from pathlib import Path
import tempfile
import pandas as pd

from seam.analysis.plotting import (
    plot_contamination_propagation,
    plot_memory_collapse,
    plot_performance_comparison,
)


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

        assert p1.exists()
        assert p2.exists()
        assert p3.exists()
