"""Analyze bargaining metrics to clarify the distinction between Fairness, Acceptance Rate, and Welfare."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def analyze_bargaining_runs() -> dict[str, Any]:
    runs_dir = Path("runs/experiments/bargaining_game")
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Extract event logs to compute acceptance rates, total welfare, and fairness
    records = []
    for run_folder in runs_dir.iterdir():
        if not run_folder.is_dir():
            continue
        events_file = run_folder / "events.jsonl"
        summary_file = run_folder / "summary.json"
        if not events_file.exists() or not summary_file.exists():
            continue

        with open(summary_file, encoding="utf-8") as f:
            summary_data = json.load(f)

        # Run folders are named such as exp_<policy>_<topology>_<poisoning>_seed<N>_<timestamp>.
        # Inspect events to derive acceptance, welfare, and proposer-share statistics.
        total_rounds = 0
        accepted_deals = 0
        total_proposer_shares = []
        total_welfare = 0.0

        with open(events_file, encoding="utf-8") as f:
            seen_rounds = set()
            for line in f:
                ev = json.loads(line)
                rnd = ev.get("round")
                deal_info = ev.get("info", {}).get("deal", {})
                if rnd not in seen_rounds and deal_info:
                    seen_rounds.add(rnd)
                    total_rounds += 1
                    if deal_info.get("accepted"):
                        accepted_deals += 1
                        split = deal_info.get("proposed_split", [50, 50])
                        total_proposer_shares.append(split[0])
                        total_welfare += sum(split)

        acceptance_rate = accepted_deals / max(1, total_rounds)
        fairness_score = summary_data.get("final_score", 0.0)

        # Policy & Topology from metadata or directory name
        run_name = run_folder.name
        if "structured_incremental" in run_name:
            policy = "structured_incremental"
        elif "raw_trajectory_buffer" in run_name:
            policy = "raw_trajectory_buffer"
        else:
            policy = "naive_overwrite"

        if "full_broadcast" in run_name:
            topology = "full_broadcast"
        elif "ring" in run_name:
            topology = "ring"
        else:
            topology = "off"

        poisoning = "internal" if "internal" in run_name else "clean"

        records.append(
            {
                "run_id": run_name,
                "policy": policy,
                "topology": topology,
                "poisoning": poisoning,
                "fairness_score": fairness_score,
                "acceptance_rate": acceptance_rate,
                "accepted_deals": accepted_deals,
                "total_welfare": total_welfare,
                "mean_proposer_share": np.mean(total_proposer_shares)
                if total_proposer_shares
                else 0.0,
            }
        )

    bdf = pd.DataFrame(records)
    bdf.to_csv(reports_dir / "bargaining_detailed_metrics.csv", index=False)

    # Group by condition
    grouped = (
        bdf.groupby(["policy", "topology", "poisoning"])
        .agg(
            mean_fairness=("fairness_score", "mean"),
            std_fairness=("fairness_score", "std"),
            mean_acceptance=("acceptance_rate", "mean"),
            std_acceptance=("acceptance_rate", "std"),
            mean_welfare=("total_welfare", "mean"),
            count=("run_id", "count"),
        )
        .reset_index()
    )

    # Generate Markdown Report
    md_lines = [
        "# Game-Theoretic Clarification of Bargaining Metrics",
        "",
        "## Motivation & Formal Distinction",
        "In multi-agent bargaining literature, a common source of reviewer confusion is conflating **Fairness** with **Efficiency (Welfare)** or **Agreement Frequency (Acceptance Rate)**.",
        "",
        "In SEAM's Bargaining Environment:",
        "1. **Fairness Score** ($S_{\\text{fair}}$):",
        "   $$S_{\\text{fair}} = \\frac{1}{1 + \\frac{1}{|\\mathcal{D}|}\\sum_{(s_1, s_2) \\in \\mathcal{D}} |s_1 - 50|}$$",
        "   where $\\mathcal{D}$ is the set of accepted deals (pie size = 100). If no deal is struck ($|\\mathcal{D}| = 0$), $S_{\\text{fair}} = 0.0$.",
        "2. **Acceptance Rate** ($\\alpha$):",
        "   $$\\alpha = \\frac{|\\mathcal{D}|}{T}$$",
        "   Measuring negotiation deadlock vs. convergence.",
        "3. **Total Welfare** ($W$):",
        "   $$W = \\sum_{(s_1, s_2) \\in \\mathcal{D}} (s_1 + s_2) = 100 \\times |\\mathcal{D}|$$",
        "",
        "### Key Insight:",
        "An accepted proposal of `99/1` achieves maximal welfare ($100$ per round) but exhibits minimal fairness ($S_{\\text{fair}} = 1/(1+49) = 0.02$). Conversely, reaching a single `50/50` agreement across 20 rounds achieves perfect conditional fairness ($1.0$) but low total welfare. Therefore, SEAM reports **Fairness Score** to measure whether self-evolving reflection helps agents converge on the egalitarian Nash bargaining solution rather than greedy unilateral exploitation.",
        "",
        "## Empirical Comparison Across Policies & Topologies",
        "",
        "| Policy | Topology | Poisoning | Mean Fairness Score | Mean Acceptance Rate | Mean Total Welfare | Runs ($N$) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]

    for _, row in grouped.iterrows():
        md_lines.append(
            f"| `{row['policy']}` | `{row['topology']}` | `{row['poisoning']}` | {row['mean_fairness']:.4f} ± {row['std_fairness']:.3f} | {row['mean_acceptance'] * 100:.1f}% ± {row['std_acceptance'] * 100:.1f}% | {row['mean_welfare']:.1f} | {int(row['count'])} |"
        )

    md_lines.extend(
        [
            "",
            "## Recommendations for Manuscript Writing",
            "- Always refer to this metric as **Fairness Score** (or *Egalitarian Fairness Index*), never generic 'performance' or 'reward'.",
            "- Note in the Methods section that when negotiations completely deadlock ($0$ agreements), Fairness Score is safely bounded at $0.0$.",
            "- Explicitly contrast with Resource Foraging (which measures environmental extraction efficiency) and Number Guessing (which measures coordination speed $1/T$).",
        ]
    )

    with open(reports_dir / "bargaining_metric_clarification.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(
        f"Generated bargaining clarification report at {reports_dir / 'bargaining_metric_clarification.md'}"
    )
    return {"bdf_len": len(bdf), "grouped_len": len(grouped)}


if __name__ == "__main__":
    analyze_bargaining_runs()
