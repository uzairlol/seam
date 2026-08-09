"""ResultAggregator for processing multi-run experiment results into summary tables and statistics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

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
        """Group results by (policy, topology, poisoning_mode) and compute summary statistics.

        Returns:
            DataFrame with aggregated metrics, including mean, std, count, SEM, and 95% CI.
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

        # Compute Standard Error of Mean (SEM = std / sqrt(n)) and 95% CI using a t distribution.
        for m in target_metrics:
            std_col = f"{m}_std"
            count_col = f"{m}_count"
            mean_col = f"{m}_mean"
            if std_col in grouped.columns and count_col in grouped.columns and mean_col in grouped.columns:
                counts = np.maximum(1, grouped[count_col].astype(float))
                std = grouped[std_col].astype(float)
                mean = grouped[mean_col].astype(float)
                sem = std / np.sqrt(counts)
                grouped[f"{m}_sem"] = sem

                ci_half_width = pd.Series(0.0, index=grouped.index, dtype=float)
                valid_mask = counts > 1
                if valid_mask.any():
                    dfree = counts[valid_mask] - 1
                    critical_values = stats.t.ppf(0.975, dfree)
                    ci_half_width.loc[valid_mask] = critical_values * sem[valid_mask]

                grouped[f"{m}_ci_low"] = np.clip(mean - ci_half_width, 0.0, 1.0)
                grouped[f"{m}_ci_high"] = np.clip(mean + ci_half_width, 0.0, 1.0)

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
            "| Policy | Topology | Poisoning | Score (Mean ± 95% CI) | Self-BLEU (Mean ± 95% CI) | Contamination Rate (Mean ± 95% CI) | Runs |",
            "|---|---|---|---|---|---|---|",
        ]

        for _, row in agg_df.iterrows():
            pol = row.get("policy", "N/A")
            top = row.get("topology", "N/A")
            poi = row.get("poisoning_mode", "N/A")

            score_m = row.get("final_score_mean", 0.0)
            score_lo = row.get("final_score_ci_low", score_m)
            score_hi = row.get("final_score_ci_high", score_m)
            score_str = f"{score_m:.4f} [{score_lo:.4f}, {score_hi:.4f}]"

            bleu_m = row.get("mean_self_bleu_mean", 0.0)
            bleu_lo = row.get("mean_self_bleu_ci_low", bleu_m)
            bleu_hi = row.get("mean_self_bleu_ci_high", bleu_m)
            bleu_str = f"{bleu_m:.4f} [{bleu_lo:.4f}, {bleu_hi:.4f}]"

            cont_m = row.get("peer_contamination_rate_mean", 0.0)
            cont_lo = row.get("peer_contamination_rate_ci_low", cont_m)
            cont_hi = row.get("peer_contamination_rate_ci_high", cont_m)
            n_runs = int(row.get("final_score_count", 0))

            cont_str = f"{cont_m:.2%} [{cont_lo:.2%}, {cont_hi:.2%}]"

            lines.append(f"| {pol} | {top} | {poi} | {score_str} | {bleu_str} | {cont_str} | {n_runs} |")

        return "\n".join(lines)
