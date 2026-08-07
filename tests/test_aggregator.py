"""Unit tests for ResultAggregator in seam.analysis.aggregator."""

from __future__ import annotations

from pathlib import Path
import tempfile
import pandas as pd
import pytest

from seam.analysis.aggregator import ResultAggregator


def test_aggregator_with_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        df_sample = pd.DataFrame([
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
        ])
        df_sample.to_csv(tmp_path / "results_summary.csv", index=False)

        agg = ResultAggregator(tmp_path)
        summary_df = agg.aggregate_conditions()

        assert len(summary_df) == 1
        assert pytest.approx(summary_df.iloc[0]["final_score_mean"]) == 0.85
        assert "final_score_sem" in summary_df.columns

        md_table = agg.to_markdown_table()
        assert "| naive_overwrite | ring | clean |" in md_table
