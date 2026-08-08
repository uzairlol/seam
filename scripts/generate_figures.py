"""CLI script to generate publication-ready plots and statistical tables from experiment outputs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from seam.analysis.aggregator import ResultAggregator
from seam.analysis.plotting import (
    plot_contamination_propagation,
    plot_memory_collapse,
    plot_performance_comparison,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_figures(input_dir: str = "runs/experiments", figures_dir: str = "figures") -> None:
    """Read experiment outputs and generate plots and markdown summary table.

    Args:
        input_dir: Directory containing experiment runs / results_summary.csv.
        figures_dir: Output folder for generated PNG plots and summary table.
    """
    inp_path = Path(input_dir)
    fig_path = Path(figures_dir)
    fig_path.mkdir(parents=True, exist_ok=True)

    aggregator = ResultAggregator(inp_path)
    df = aggregator.df

    if df.empty:
        logger.warning("No experiment data found in '%s'. Run run_experiments.py first.", inp_path)
        return

    logger.info("Loaded %d run records from %s", len(df), inp_path)

    # 1. Generate Figures
    plot_performance_comparison(df, fig_path / "performance_comparison.png")
    plot_memory_collapse(df, fig_path / "memory_collapse.png")
    plot_contamination_propagation(df, fig_path / "contamination_propagation.png")

    # 2. Export Markdown Table
    stats_df = aggregator.aggregate_conditions()
    stats_csv = fig_path / "summary_statistics.csv"
    stats_df.to_csv(stats_csv, index=False)
    logger.info("Saved summary statistics to %s", stats_csv)

    table_md = aggregator.to_markdown_table()
    table_file = fig_path / "summary_table.md"
    with open(table_file, "w", encoding="utf-8") as f:
        f.write("# SEAM Experiment Metric Summary Table\n\n" + table_md + "\n")
    logger.info("Saved summary table to %s", table_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SEAM Plots and Summary Tables")
    parser.add_argument("--indir", type=str, default="runs/experiments", help="Input experiment runs directory")
    parser.add_argument("--outdir", type=str, default="figures", help="Output figures directory")
    args = parser.parse_args()

    generate_figures(input_dir=args.indir, figures_dir=args.outdir)


if __name__ == "__main__":
    main()
