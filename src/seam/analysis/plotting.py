"""Plotting engine producing publication-quality figures for performance, collapse, and contamination."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Set style globally
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
})


def plot_performance_comparison(df: pd.DataFrame, output_file: str | Path = "figures/performance_comparison.png") -> Path:
    """Generate bar chart comparing ground truth task scores across conditions.

    Args:
        df: DataFrame containing experiment results summary.
        output_file: Target PNG file path.

    Returns:
        Path object pointing to saved figure.
    """
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    if "policy" in df.columns and "final_score" in df.columns:
        hue_col = "topology" if "topology" in df.columns else None
        sns.barplot(
            data=df,
            x="policy",
            y="final_score",
            hue=hue_col,
            ax=ax,
            capsize=0.1,
            err_kws={"linewidth": 1.5},
        )
        ax.set_title("Ground Truth Task Performance by Memory Policy & Topology")
        ax.set_ylabel("Final Task Score")
        ax.set_xlabel("Memory Policy")
        ax.set_ylim(0.0, 1.0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("Saved performance comparison plot to %s", out_path)
    return out_path


def plot_memory_collapse(df: pd.DataFrame, output_file: str | Path = "figures/memory_collapse.png") -> Path:
    """Generate bar chart comparing Self-BLEU memory collapse metrics across conditions.

    Args:
        df: DataFrame containing experiment results summary.
        output_file: Target PNG file path.

    Returns:
        Path object pointing to saved figure.
    """
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    if "policy" in df.columns and "mean_self_bleu" in df.columns:
        hue_col = "topology" if "topology" in df.columns else None
        sns.barplot(
            data=df,
            x="policy",
            y="mean_self_bleu",
            hue=hue_col,
            ax=ax,
            capsize=0.1,
            err_kws={"linewidth": 1.5},
        )
        ax.set_title("Memory Collapse (Self-BLEU Metric) across Conditions")
        ax.set_ylabel("Mean Self-BLEU (Higher = Greater Collapse)")
        ax.set_xlabel("Memory Policy")
        ax.set_ylim(0.0, 1.0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("Saved memory collapse plot to %s", out_path)
    return out_path


def plot_contamination_propagation(df: pd.DataFrame, output_file: str | Path = "figures/contamination_propagation.png") -> Path:
    """Generate bar chart showing peer contamination rate across topologies under memory poisoning.

    Args:
        df: DataFrame containing experiment results summary.
        output_file: Target PNG file path.

    Returns:
        Path object pointing to saved figure.
    """
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    if "topology" in df.columns and "peer_contamination_rate" in df.columns:
        sns.barplot(
            data=df,
            x="topology",
            y="peer_contamination_rate",
            hue="poisoning_mode" if "poisoning_mode" in df.columns else None,
            ax=ax,
            capsize=0.1,
            err_kws={"linewidth": 1.5},
        )
        ax.set_title("Peer Contamination Propagation Rate by Network Topology")
        ax.set_ylabel("Peer Contamination Rate (%)")
        ax.set_xlabel("Network Sharing Topology")
        ax.set_ylim(0.0, 1.0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("Saved contamination propagation plot to %s", out_path)
    return out_path
