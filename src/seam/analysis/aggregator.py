"""ResultAggregator for processing multi-run experiment results into summary tables and statistics."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from seam.analysis.efficacy import compute_efficacy_gap
from seam.logging.rehydrator import RunRehydrator

logger = logging.getLogger(__name__)


class ResultAggregator:
    """Aggregates experiment run logs and calculates statistical summaries across seeds.

    Args:
        runs_dir: Path to directory containing run folders or summary CSV file.
    """

    def __init__(self, runs_dir: str | Path = "runs/experiments") -> None:
        self.runs_dir = Path(runs_dir)
        self.df = compute_efficacy_gap(self._load_data())

    def _load_data(self) -> pd.DataFrame:
        """Load experiment data into a pandas DataFrame.

        The directory scan is treated as the ground truth for record counts. A
        cached ``results_summary.csv`` is used only when it is at least as
        complete as the run directories it describes; if the scan finds more
        completed ``summary.json`` files than the CSV has rows, the CSV is
        stale and the freshly rehydrated records take precedence.
        """
        csv_df = self._load_summary_csv()
        scan_df = self._scan_run_dirs()

        if scan_df is None:
            return csv_df if csv_df is not None else pd.DataFrame()

        if csv_df is None or len(scan_df) > len(csv_df):
            if csv_df is not None:
                logger.warning(
                    "results_summary.csv has %d rows but %d completed run directories "
                    "were found — ignoring stale CSV and rehydrating from runs.",
                    len(csv_df),
                    len(scan_df),
                )
            return scan_df

        return csv_df

    def _load_summary_csv(self) -> pd.DataFrame | None:
        """Read ``results_summary.csv`` if present and parseable."""
        summary_csv = self.runs_dir / "results_summary.csv"
        if not summary_csv.exists():
            return None
        try:
            return pd.read_csv(summary_csv)
        except OSError as exc:
            logger.warning("Could not read %s: %s", summary_csv, exc)
            return None

    def _scan_run_dirs(self) -> pd.DataFrame | None:
        """Recursively rehydrate ``summary.json`` files from run subdirectories."""
        if not self.runs_dir.exists():
            return None

        records = []
        for summary_json in self.runs_dir.rglob("summary.json"):
            run_dir = summary_json.parent
            try:
                rehydrator = RunRehydrator(run_dir)
                meta = rehydrator.load_metadata()
                summary = rehydrator.load_summary()
                summary_info = summary.get("summary_info", {})

                topo = (
                    meta.get("topology")
                    if meta.get("topology") is not None
                    else meta.get("sharing_mode")
                )
                records.append(
                    {
                        "run_id": meta.get("run_id"),
                        "experiment_id": meta.get("experiment_id"),
                        "policy": meta.get("memory_policy"),
                        "topology": topo,
                        "poisoning_mode": meta.get("poisoning_mode"),
                        "seed": meta.get("seed"),
                        "final_score": summary.get("final_score", 0.0),
                        "oracle_score": summary_info.get("oracle_score"),
                        "mean_self_bleu": summary_info.get("mean_self_bleu", 0.0),
                        "peer_contamination_rate": summary_info.get("peer_contamination_rate", 0.0),
                        "poison_dosage_rate": summary_info.get("poison_dosage_rate", 0.0),
                        "propagation_latency": summary_info.get("propagation_latency"),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to rehydrate %s: %s", run_dir, exc)

        return pd.DataFrame(records) if records else None

    def aggregate_conditions(self) -> pd.DataFrame:
        """Group results by (policy, topology, poisoning_mode) and compute summary statistics.

        Returns:
            DataFrame with aggregated metrics, including mean, std, count, SEM, and 95% CI.
        """
        if self.df.empty:
            return pd.DataFrame()

        group_cols = [
            col for col in ["policy", "topology", "poisoning_mode"] if col in self.df.columns
        ]
        if not group_cols:
            return pd.DataFrame()

        metrics = ["final_score", "mean_self_bleu", "peer_contamination_rate"]
        target_metrics = [m for m in metrics if m in self.df.columns]
        for optional in ("poison_dosage_rate", "oracle_score", "efficacy_gap"):
            if optional in self.df.columns:
                target_metrics.append(optional)

        agg_dict: dict[str, list[str]] = {m: ["mean", "std", "count"] for m in target_metrics}
        grouped = self.df.groupby(group_cols).agg(agg_dict)

        # Flatten multi-level columns
        flat_columns: list[str] = []
        for col in grouped.columns:
            if isinstance(col, tuple):
                flat_columns.append("_".join(str(part) for part in col))
            else:
                flat_columns.append(str(col))
        grouped.columns = flat_columns
        grouped = grouped.reset_index()

        # Compute Standard Error of Mean (SEM = std / sqrt(n)) and 95% CI using a t distribution.
        for m in target_metrics:
            std_col = f"{m}_std"
            count_col = f"{m}_count"
            mean_col = f"{m}_mean"
            if (
                std_col in grouped.columns
                and count_col in grouped.columns
                and mean_col in grouped.columns
            ):
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

                if m in {
                    "final_score",
                    "mean_self_bleu",
                    "peer_contamination_rate",
                    "poison_dosage_rate",
                    "oracle_score",
                }:
                    grouped[f"{m}_ci_low"] = np.clip(mean - ci_half_width, 0.0, 1.0)
                    grouped[f"{m}_ci_high"] = np.clip(mean + ci_half_width, 0.0, 1.0)
                else:
                    grouped[f"{m}_ci_low"] = mean - ci_half_width
                    grouped[f"{m}_ci_high"] = mean + ci_half_width

        return grouped

    def to_markdown_table(self) -> str:
        """Format aggregated metrics into a GitHub-flavored Markdown table.

        Returns:
            Markdown formatted table string.
        """
        agg_df = self.aggregate_conditions()
        if agg_df.empty:
            return "No data available."

        has_oracle = "oracle_score_mean" in agg_df.columns
        has_efficacy = "efficacy_gap_mean" in agg_df.columns

        header = [
            "Policy",
            "Topology",
            "Poisoning",
            "Score (Mean ± 95% CI)",
            "Self-BLEU (Mean ± 95% CI)",
            "Contamination Rate (Mean ± 95% CI)",
            "Runs",
        ]
        if has_oracle:
            header.append("Oracle Score (Mean ± 95% CI)")
        if has_efficacy:
            header.append("Efficacy Gap (Mean ± 95% CI)")
        lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

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

            row_parts = [pol, top, poi, score_str, bleu_str, cont_str, str(n_runs)]
            if has_oracle:
                oracle_m = row.get("oracle_score_mean", 0.0)
                oracle_lo = row.get("oracle_score_ci_low", oracle_m)
                oracle_hi = row.get("oracle_score_ci_high", oracle_m)
                row_parts.append(f"{oracle_m:.4f} [{oracle_lo:.4f}, {oracle_hi:.4f}]")
            if has_efficacy:
                eff_m = row.get("efficacy_gap_mean", 0.0)
                eff_lo = row.get("efficacy_gap_ci_low", eff_m)
                eff_hi = row.get("efficacy_gap_ci_high", eff_m)
                row_parts.append(f"{eff_m:.3f} [{eff_lo:.3f}, {eff_hi:.3f}]")

            lines.append("| " + " | ".join(row_parts) + " |")

        return "\n".join(lines)
