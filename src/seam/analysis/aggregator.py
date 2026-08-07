"""ResultAggregator for processing multi-run experiment results into summary tables and statistics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from seam.logging.rehydrator import RunRehydrator

logger = logging.getLogger(__name__)


class ResultAggregator:
    """Aggregates experiment run logs and calculates statistical summaries across seeds.

    Args:
        runs_dir: Path to directory containing run folders or summary CSV file.
    """

    def __init__(self, runs_dir: str | Path = "runs/experiments") -> None:
        self.runs_dir = Path(runs_dir)
        self.df = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """Load experiment data into a pandas DataFrame."""
        summary_csv = self.runs_dir / "results_summary.csv"
        if summary_csv.exists():
            try:
                return pd.read_csv(summary_csv)
            except Exception as exc:
                logger.warning("Could not read %s: %s — scanning subdirectories", summary_csv, exc)

        # Fallback: scan subdirectories for summary.json files
        records = []
        if self.runs_dir.exists():
            for child in self.runs_dir.glob("*"):
                if child.is_dir():
                    summary_json = child / "summary.json"
                    if summary_json.exists():
                        try:
                            rehydrator = RunRehydrator(child)
                            meta = rehydrator.load_metadata()
                            summary = rehydrator.load_summary()
                            summary_info = summary.get("summary_info", {})

                            records.append({
                                "run_id": meta.get("run_id"),
                                "experiment_id": meta.get("experiment_id"),
                                "policy": meta.get("memory_policy"),
                                "topology": meta.get("sharing_mode"),
                                "poisoning_mode": meta.get("poisoning_mode"),
                                "seed": meta.get("seed"),
                                "final_score": summary.get("final_score", 0.0),
                                "mean_self_bleu": summary_info.get("mean_self_bleu", 0.0),
                                "peer_contamination_rate": summary_info.get("peer_contamination_rate", 0.0),
                            })
                        except Exception as exc:
                            logger.debug("Failed to rehydrate %s: %s", child, exc)

        return pd.DataFrame(records)

    def aggregate_conditions(self) -> pd.DataFrame:
        """Group results by (policy, topology, poisoning_mode) and compute mean and SEM.

        Returns:
            DataFrame with aggregated metrics.
        """
        if self.df.empty:
            return pd.DataFrame()

        group_cols = [col for col in ["policy", "topology", "poisoning_mode"] if col in self.df.columns]
        if not group_cols:
            return pd.DataFrame()

        metrics = ["final_score", "mean_self_bleu", "peer_contamination_rate"]
        target_metrics = [m for m in metrics if m in self.df.columns]

        agg_dict: dict[str, list[str]] = {m: ["mean", "std", "count"] for m in target_metrics}
        grouped = self.df.groupby(group_cols).agg(agg_dict)

        # Flatten multi-level columns
        grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]
        grouped = grouped.reset_index()

        # Compute Standard Error of Mean (SEM = std / sqrt(n))
        for m in target_metrics:
            if f"{m}_std" in grouped.columns and f"{m}_count" in grouped.columns:
                grouped[f"{m}_sem"] = grouped[f"{m}_std"] / np.sqrt(np.maximum(1, grouped[f"{m}_count"]))

        return grouped

    def to_markdown_table(self) -> str:
        """Format aggregated metrics into a GitHub-flavored Markdown table.

        Returns:
            Markdown formatted table string.
        """
        agg_df = self.aggregate_conditions()
        if agg_df.empty:
            return "No data available."

        lines = [
            "| Policy | Topology | Poisoning | Score (Mean ± SEM) | Self-BLEU (Mean ± SEM) | Contamination Rate | Runs |",
            "|---|---|---|---|---|---|---|",
        ]

        for _, row in agg_df.iterrows():
            pol = row.get("policy", "N/A")
            top = row.get("topology", "N/A")
            poi = row.get("poisoning_mode", "N/A")

            score_m = row.get("final_score_mean", 0.0)
            score_s = row.get("final_score_sem", 0.0)
            score_str = f"{score_m:.4f} ± {score_s:.4f}"

            bleu_m = row.get("mean_self_bleu_mean", 0.0)
            bleu_s = row.get("mean_self_bleu_sem", 0.0)
            bleu_str = f"{bleu_m:.4f} ± {bleu_s:.4f}"

            cont_m = row.get("peer_contamination_rate_mean", 0.0)
            n_runs = int(row.get("final_score_count", 0))

            lines.append(f"| {pol} | {top} | {poi} | {score_str} | {bleu_str} | {cont_m:.2%} | {n_runs} |")

        return "\n".join(lines)
