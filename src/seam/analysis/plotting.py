"""Plotting engine producing publication-quality figures for performance, collapse, and contamination."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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

_METRICS = ("final_score", "mean_self_bleu", "peer_contamination_rate")


def _usable_data(
    df: pd.DataFrame,
    required: list[str],
    numeric: list[str],
) -> pd.DataFrame:
    """Return rows usable for a plot, or an empty frame when inputs are incomplete."""
    missing = [column for column in required if column not in df.columns]
    if missing:
        logger.warning("Skipping plot data; missing columns: %s", missing)
        return pd.DataFrame()

    result = df.copy()
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.dropna(subset=required + numeric)


def _start_figure(output_file: str | Path, figsize: tuple[float, float]) -> tuple[Path, plt.Figure, plt.Axes]:
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    return out_path, fig, ax


def _finish_figure(fig: plt.Figure, out_path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot to %s", out_path)
    return out_path


def _show_no_data(ax: plt.Axes, title: str) -> None:
    ax.text(0.5, 0.5, "No usable data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_axis_off()


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
        ax.set_ylabel("Peer Contamination Rate (proportion)")
        ax.set_xlabel("Network Sharing Topology")
        ax.set_ylim(0.0, 1.0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("Saved contamination propagation plot to %s", out_path)
    return out_path


def plot_metric_condition_heatmap(
    df: pd.DataFrame,
    output_file: str | Path = "figures/metric_condition_heatmap.png",
    metric: str = "final_score",
) -> Path:
    """Plot condition means as a policy-by-topology heatmap."""
    out_path, fig, ax = _start_figure(output_file, (8, 5))
    title = f"{metric.replace('_', ' ').title()} by Memory Policy and Topology"
    data = _usable_data(df, ["policy", "topology", metric], [metric]) if metric in _METRICS else pd.DataFrame()
    if data.empty:
        _show_no_data(ax, title)
    else:
        table = data.pivot_table(index="policy", columns="topology", values=metric, aggfunc="mean")
        sns.heatmap(table, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Network Topology")
        ax.set_ylabel("Memory Policy")
    return _finish_figure(fig, out_path)


def plot_poisoning_performance_comparison(
    df: pd.DataFrame,
    output_file: str | Path = "figures/poisoning_performance_comparison.png",
) -> Path:
    """Compare run-level task scores under clean and poisoned conditions."""
    out_path, fig, ax = _start_figure(output_file, (9, 5))
    data = _usable_data(df, ["poisoning_mode", "final_score"], ["final_score"])
    if data.empty:
        _show_no_data(ax, "Task Performance by Poisoning Condition")
    else:
        hue = "policy" if "policy" in data.columns and data["policy"].nunique() > 1 else None
        sns.stripplot(
            data=data,
            x="poisoning_mode",
            y="final_score",
            hue=hue,
            dodge=hue is not None,
            alpha=0.65,
            jitter=0.12,
            ax=ax,
        )
        sns.pointplot(
            data=data,
            x="poisoning_mode",
            y="final_score",
            hue=hue,
            dodge=0.35 if hue is not None else 0,
            errorbar=("ci", 95),
            join=False,
            markers="D",
            color="black",
            ax=ax,
        )
        if hue is not None:
            handles, labels = ax.get_legend_handles_labels()
            unique = data[hue].nunique()
            ax.legend(handles[:unique], labels[:unique], title=hue)
        ax.set_title("Task Performance by Poisoning Condition")
        ax.set_xlabel("Poisoning Mode")
        ax.set_ylabel("Final Task Score")
        ax.set_ylim(0, 1)
    return _finish_figure(fig, out_path)


def plot_collapse_performance_tradeoff(
    df: pd.DataFrame,
    output_file: str | Path = "figures/collapse_performance_tradeoff.png",
) -> Path:
    """Plot the descriptive association between Self-BLEU and task score."""
    out_path, fig, ax = _start_figure(output_file, (8, 5))
    data = _usable_data(df, ["mean_self_bleu", "final_score"], ["mean_self_bleu", "final_score"])
    if data.empty:
        _show_no_data(ax, "Memory Collapse versus Task Performance")
    else:
        kwargs = {"data": data, "x": "mean_self_bleu", "y": "final_score", "ax": ax}
        if "policy" in data.columns:
            kwargs["hue"] = "policy"
        if "topology" in data.columns:
            kwargs["style"] = "topology"
        sns.scatterplot(**kwargs)
        ax.set_title("Memory Collapse versus Task Performance")
        ax.set_xlabel("Mean Self-BLEU")
        ax.set_ylabel("Final Task Score")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    return _finish_figure(fig, out_path)


def plot_run_level_distributions(
    df: pd.DataFrame,
    output_file: str | Path = "figures/run_level_distributions.png",
) -> Path:
    """Show seed/run-level distributions for the three primary metrics."""
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [metric for metric in _METRICS if metric in df.columns]
    fig, axes = plt.subplots(1, max(1, len(metrics)), figsize=(5 * max(1, len(metrics)), 5), squeeze=False)
    axes = axes[0]
    if not metrics or "policy" not in df.columns:
        _show_no_data(axes[0], "Run-Level Metric Distributions")
    else:
        for ax, metric in zip(axes, metrics):
            data = _usable_data(df, ["policy", metric], [metric])
            if data.empty:
                _show_no_data(ax, metric.replace("_", " ").title())
                continue
            sns.boxplot(data=data, x="policy", y=metric, color="#9ecae1", ax=ax)
            sns.stripplot(data=data, x="policy", y=metric, color="black", alpha=0.55, jitter=0.12, ax=ax)
            ax.set_title(metric.replace("_", " ").title())
            ax.set_xlabel("Memory Policy")
            ax.set_ylabel("Value")
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Run-Level Metric Distributions")
    return _finish_figure(fig, out_path)
