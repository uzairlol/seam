"""Generate figures and summary statistics for Phase 5 single-agent baseline runs.

Baseline runs are discovered dynamically from ``runs/baselines`` (or an
``--indir`` override).  Each run directory contributes ``metadata.json`` +
``summary.json`` via :class:`RunRehydrator`.  Because the Phase-5 runner writes
all three environments into one flat folder, this script keys on the
``env_type`` recorded in each run's metadata rather than the directory layout.

The script reuses the existing analysis stack (``seam.analysis.efficacy``,
``seam.analysis.significance`` and ``seam.analysis.plotting``) so the figures
are consistent with the experiment-tier analysis in ``generate_figures.py``.

Outputs (defaults):
    * ``figures/baselines/<env>/``       per-environment PNG plots
    * ``figures/baselines/``             cross-environment comparisons
    * ``reports/baselines/summary_statistics.csv``
    * ``reports/baselines/summary_table.md``
    * ``reports/baselines/statistical_tests.csv``
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from seam.analysis.efficacy import compute_efficacy_gap
from seam.analysis.plotting import (
    plot_collapse_performance_tradeoff,
    plot_memory_collapse,
    plot_performance_comparison,
    plot_run_level_distributions,
)
from seam.analysis.significance import compute_cliffs_delta, interpret_cliffs_delta
from seam.logging.rehydrator import RunRehydrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

POLICY_ORDER = ["no_memory", "naive_overwrite", "raw_trajectory_buffer", "structured_incremental"]
ENV_ORDER = ["resource_foraging", "number_guessing", "bargaining_game"]
METRICS = ["final_score", "mean_self_bleu", "oracle_score"]
# Metrics whose natural domain is [0, 1]; efficacy_gap may exceed 1 when a
# policy outperforms the deterministic oracle reference, so its CI is unclipped.
NATIVE_ZERO_ONE = {"final_score", "mean_self_bleu", "oracle_score"}


@dataclass
class TestRow:
    environment: str
    test_id: str
    metric: str
    condition_a: str
    condition_b: str
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    median_a: float
    median_b: float
    u_stat: float
    p_value: float
    p_value_fdr: float
    cliffs_delta: float
    effect_size_interpretation: str
    is_statistically_significant: bool


def _load_data(indir: Path) -> pd.DataFrame:
    """Rehydrate every baseline ``summary.json`` into a flat long-form frame."""
    records = []
    for run_dir in sorted(indir.rglob("summary.json")):
        try:
            rehydrator = RunRehydrator(run_dir.parent)
            meta = rehydrator.load_metadata()
            summary = rehydrator.load_summary()
            summary_info = summary.get("summary_info", {})
            # Baselines are single-agent / isolated: the *_effective* topology is
            # the sharing mode ("off"); metadata defaults topology to
            # "full_broadcast" even though sharing is disabled.
            topology = meta.get("sharing_mode") or meta.get("topology")
            records.append(
                {
                    "run_id": meta.get("run_id"),
                    "environment": meta.get("env_type"),
                    "policy": meta.get("memory_policy"),
                    "topology": topology,
                    "poisoning_mode": meta.get("poisoning_mode"),
                    "seed": meta.get("seed"),
                    "rounds_played": summary_info.get("rounds_played"),
                    "final_score": summary.get("final_score", 0.0),
                    "oracle_score": summary_info.get("oracle_score"),
                    "mean_self_bleu": summary_info.get("mean_self_bleu", 0.0),
                    "peer_contamination_rate": summary_info.get("peer_contamination_rate", 0.0),
                    "poison_dosage_rate": summary_info.get("poison_dosage_rate", 0.0),
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to rehydrate %s: %s", run_dir.parent, exc)
    return pd.DataFrame(records)


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-environment efficacy gap and a stable policy order."""
    if df.empty:
        return df
    enriched = pd.DataFrame(index=df.index)
    for _env, env_df in df.assign(_env=df["environment"]).groupby("_env", sort=True):
        gap = compute_efficacy_gap(env_df.drop(columns=["_env"]))
        enriched.loc[gap.index, "efficacy_gap"] = gap["efficacy_gap"]
    result = df.copy()
    result["efficacy_gap"] = enriched["efficacy_gap"]
    result["policy"] = pd.Categorical(result["policy"], categories=POLICY_ORDER, ordered=True)
    return result


def _sem_ci(values: pd.Series, confidence: float = 0.95) -> tuple[float, float]:
    """Return (sem, ci_half_width) for a t-based confidence interval."""
    values = values.dropna().astype(float)
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    sem = float(values.std(ddof=1) / np.sqrt(n))
    if n <= 1:
        return sem, 0.0
    critical = float(stats.t.ppf(1 - (1 - confidence) / 2, n - 1))
    return sem, critical * sem


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SEAM Phase 5 single-agent baselines")
    parser.add_argument("--indir", type=str, default="runs/baselines", help="Baseline runs dir")
    parser.add_argument(
        "--figdir", type=str, default="figures/baselines", help="Output figures directory"
    )
    parser.add_argument(
        "--outdir", type=str, default="reports/baselines", help="Output reports directory"
    )
    args = parser.parse_args()

    indir = Path(args.indir)
    figdir = Path(args.figdir)
    outdir = Path(args.outdir)
    figdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    df = _load_data(indir)
    if df.empty:
        logger.error("No baseline run records found under %s", indir)
        return
    df = _enrich(df)
    logger.info(
        "Loaded %d baseline records (%d environments)", len(df), df["environment"].nunique()
    )

    # 1. Summary statistics (mean / std / SEM / 95% CI) per environment x policy
    rows = []
    metrics_stats = ["final_score", "efficacy_gap", "mean_self_bleu", "oracle_score"]
    for env, env_df in df.groupby("environment", sort=True):
        for policy, pol_df in env_df.groupby("policy", observed=True, sort=False):
            record: dict[str, object] = {
                "environment": env,
                "policy": policy,
                "n_runs": int(len(pol_df)),
            }
            for metric in metrics_stats:
                values = pd.to_numeric(pol_df[metric], errors="coerce")
                if values.notna().sum() == 0:
                    continue
                mean = float(values.mean())
                std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                sem, ci_half = _sem_ci(values)
                record[f"{metric}_mean"] = mean
                record[f"{metric}_std"] = std
                record[f"{metric}_sem"] = sem
                if metric in NATIVE_ZERO_ONE:
                    record[f"{metric}_ci_low"] = float(max(mean - ci_half, 0.0))
                    record[f"{metric}_ci_high"] = float(min(mean + ci_half, 1.0))
                else:
                    record[f"{metric}_ci_low"] = float(mean - ci_half)
                    record[f"{metric}_ci_high"] = float(mean + ci_half)
            record["peer_contamination_rate_mean"] = float(pol_df["peer_contamination_rate"].mean())
            record["poison_dosage_rate_mean"] = float(pol_df["poison_dosage_rate"].mean())
            rows.append(record)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(outdir / "summary_statistics.csv", index=False)
    logger.info("Saved summary statistics to %s", outdir / "summary_statistics.csv")

    # 2. Statistical tests: active policies vs no_memory control, plus
    #    structured vs naive, on final_score and efficacy_gap.
    control = "no_memory"
    comparisons = [
        ("structured_incremental", control),
        ("raw_trajectory_buffer", control),
        ("naive_overwrite", control),
        ("structured_incremental", "naive_overwrite"),
    ]
    gap_comparisons = [
        ("structured_incremental", "raw_trajectory_buffer"),
        ("raw_trajectory_buffer", "naive_overwrite"),
    ]
    test_rows: list[TestRow] = []
    for env, env_df in df.groupby("environment", sort=True):
        for policy_a, policy_b in comparisons:
            vals_a = env_df.loc[env_df["policy"] == policy_a, "final_score"].dropna().to_numpy()
            vals_b = env_df.loc[env_df["policy"] == policy_b, "final_score"].dropna().to_numpy()
            if len(vals_a) > 0 and len(vals_b) > 0:
                test_rows.append(
                    _mann_whitney(env, "final_score", policy_a, vals_a, policy_b, vals_b)
                )
        for policy_a, policy_b in gap_comparisons:
            vals_a = env_df.loc[env_df["policy"] == policy_a, "efficacy_gap"].dropna().to_numpy()
            vals_b = env_df.loc[env_df["policy"] == policy_b, "efficacy_gap"].dropna().to_numpy()
            if len(vals_a) > 0 and len(vals_b) > 0:
                test_rows.append(
                    _mann_whitney(env, "efficacy_gap", policy_a, vals_a, policy_b, vals_b)
                )
    test_rows = _apply_fdr(test_rows)
    tests_df = pd.DataFrame([vars(r) for r in test_rows])
    tests_df.to_csv(outdir / "statistical_tests.csv", index=False)
    logger.info(
        "Saved %d statistical tests to %s", len(test_rows), outdir / "statistical_tests.csv"
    )

    # 3. Per-environment figures (reuse experiment-tier plotting functions)
    for env in ENV_ORDER:
        env_df = df[df["environment"] == env]
        if env_df.empty:
            continue
        env_figdir = figdir / env
        env_figdir.mkdir(parents=True, exist_ok=True)
        plot_performance_comparison(env_df, env_figdir / "performance_comparison.png")
        plot_memory_collapse(env_df, env_figdir / "memory_collapse.png")
        plot_run_level_distributions(env_df, env_figdir / "run_level_distributions.png")
        plot_collapse_performance_tradeoff(env_df, env_figdir / "collapse_performance_tradeoff.png")
        _plot_efficacy_gap(env_df, env_figdir / "efficacy_gap.png")
        logger.info("Generated figures for %s", env)

    # 4. Cross-environment figures
    _plot_cross_environment(df, figdir / "cross_environment_scores.png", metric="final_score")
    _plot_cross_environment(
        df, figdir / "cross_environment_efficacy_gap.png", metric="efficacy_gap"
    )

    # 5. Markdown summary table
    _write_markdown_table(summary_df, outdir / "summary_table.md")
    logger.info("All baseline analysis outputs written under %s and %s", figdir, outdir)


def _mann_whitney(
    env: str,
    metric: str,
    policy_a: str,
    vals_a: np.ndarray,
    policy_b: str,
    vals_b: np.ndarray,
) -> TestRow:
    try:
        stat, p_value = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
    except ValueError:
        stat, p_value = 0.0, 1.0
    delta = compute_cliffs_delta(vals_a, vals_b)
    return TestRow(
        environment=env,
        test_id=f"{metric}_a={policy_a}_b={policy_b}",
        metric=metric,
        condition_a=policy_a,
        condition_b=policy_b,
        n_a=int(len(vals_a)),
        n_b=int(len(vals_b)),
        mean_a=float(np.mean(vals_a)),
        mean_b=float(np.mean(vals_b)),
        median_a=float(np.median(vals_a)),
        median_b=float(np.median(vals_b)),
        u_stat=float(stat),
        p_value=float(p_value),
        p_value_fdr=float(p_value),
        cliffs_delta=float(delta),
        effect_size_interpretation=interpret_cliffs_delta(delta),
        is_statistically_significant=bool(p_value < 0.05),
    )


def _apply_fdr(rows: list[TestRow]) -> list[TestRow]:
    """Benjamini-Hochberg FDR correction across all tests."""
    if not rows:
        return rows
    raw_p = np.array([r.p_value for r in rows])
    order = np.argsort(raw_p)
    sorted_p = raw_p[order]
    n = len(raw_p)
    q = np.empty(n)
    running_min = 1.0
    for i in range(n - 1, -1, -1):
        running_min = min(running_min, sorted_p[i] * n / (i + 1))
        q[i] = running_min
    fdr_p = np.empty(n)
    fdr_p[order] = np.clip(q, 0.0, 1.0)
    for row, corrected in zip(rows, fdr_p, strict=True):
        row.p_value_fdr = float(corrected)
        row.is_statistically_significant = bool(corrected < 0.05)
    return rows


def _plot_efficacy_gap(env_df: pd.DataFrame, output_file: Path) -> None:
    """Bar + seed-strip chart of per-policy efficacy gap for one environment."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8, 5))
    data = env_df.dropna(subset=["efficacy_gap"]).copy()
    sns.stripplot(
        data=data,
        x="policy",
        y="efficacy_gap",
        order=POLICY_ORDER,
        color="black",
        alpha=0.55,
        jitter=0.12,
        ax=ax,
    )
    sns.pointplot(
        data=data,
        x="policy",
        y="efficacy_gap",
        order=POLICY_ORDER,
        errorbar=("ci", 95),
        linestyle="none",
        markers="D",
        color="C3",
        ax=ax,
    )
    ax.set_title("Efficacy Gap (fraction of no_memory→oracle headroom recovered)")
    ax.set_xlabel("Memory Policy")
    ax.set_ylabel("Efficacy Gap")
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved efficacy gap plot to %s", output_file)


def _plot_cross_environment(df: pd.DataFrame, output_file: Path, metric: str) -> None:
    """One panel per environment with policy bars for *metric*."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    data = df.dropna(subset=[metric]).copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False, squeeze=False)
    for ax, env in zip(axes[0], ENV_ORDER, strict=True):
        env_data = data[data["environment"] == env]
        if env_data.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(env)
            continue
        sns.barplot(
            data=env_data,
            x="policy",
            y=metric,
            order=POLICY_ORDER,
            color="#4c72b0",
            errorbar=("ci", 95),
            capsize=0.1,
            ax=ax,
        )
        sns.stripplot(
            data=env_data,
            x="policy",
            y=metric,
            order=POLICY_ORDER,
            color="black",
            alpha=0.5,
            jitter=0.1,
            ax=ax,
        )
        ax.set_title(env)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle(metric.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved cross-environment plot to %s", output_file)


def _write_markdown_table(summary_df: pd.DataFrame, output_file: Path) -> None:
    lines = [
        "# SEAM Phase-5 Baseline Summary Table",
        "",
        "Single-agent, isolated (`off` topology), clean-trials baseline runs "
        "(qwen2.5:7b, seeds 40–49, 6 agents, 50 rounds).",
        "",
        "| Environment | Policy | Runs | Score (Mean ± 95% CI) | Oracle | Efficacy Gap (Mean ± 95% CI) | Self-BLEU (Mean ± 95% CI) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary_df.sort_values(["environment", "policy"]).iterrows():
        score = _fmt_ci(row, "final_score")
        oracle = f"{row.get('oracle_score_mean', np.nan):.4f}"
        gap = _fmt_ci(row, "efficacy_gap", default="n/a")
        bleu = _fmt_ci(row, "mean_self_bleu")
        lines.append(
            f"| {row['environment']} | {row['policy']} | {int(row['n_runs'])} | {score} | {oracle} | {gap} | {bleu} |"
        )
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_ci(row: pd.Series, metric: str, default: str = "—") -> str:
    mean = row.get(f"{metric}_mean")
    low = row.get(f"{metric}_ci_low")
    high = row.get(f"{metric}_ci_high")
    if mean is None or pd.isna(mean) or pd.isna(low) or pd.isna(high):
        return default
    return f"{mean:.4f} [{low:.4f}, {high:.4f}]"


if __name__ == "__main__":
    main()
