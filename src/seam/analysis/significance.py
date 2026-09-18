"""Module for statistical hypothesis testing, invariant auditing, and effect size calculation for SEAM."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class HypothesisTestResult:
    test_id: str
    environment: str
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


def compute_cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Cliff's Delta effect size for non-parametric comparisons."""
    n_x = len(x)
    n_y = len(y)
    if n_x == 0 or n_y == 0:
        return 0.0

    # Dominance matrix
    diff = x[:, None] - y[None, :]
    more = np.sum(diff > 0)
    less = np.sum(diff < 0)
    delta = (more - less) / (n_x * n_y)
    return float(delta)


def interpret_cliffs_delta(delta: float) -> str:
    """Standard interpretation of Cliff's delta magnitude."""
    abs_d = abs(delta)
    if abs_d < 0.147:
        return "negligible"
    elif abs_d < 0.33:
        return "small"
    elif abs_d < 0.474:
        return "medium"
    else:
        return "large"


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to support both aggregator formats."""
    df = df.copy()
    col_map = {
        "poisoning_mode": "poisoning",
        "final_score": "score",
        "peer_contamination_rate": "contamination_rate",
        "mean_self_bleu": "self_bleu",
    }
    for old_col, new_col in col_map.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]

    # If environment is not present, extract from run_id or directory pattern
    if "environment" not in df.columns:
        if "experiment_id" in df.columns:

            def infer_env(row: pd.Series) -> str:
                rid = str(row.get("run_id", ""))
                eid = str(row.get("experiment_id", ""))
                combined = f"{rid} {eid}".lower()
                if "bargaining" in combined:
                    return "bargaining_game"
                elif "number_guessing" in combined:
                    return "number_guessing"
                else:
                    return "resource_foraging"

            df["environment"] = df.apply(infer_env, axis=1)
        else:
            df["environment"] = "resource_foraging"

    return df


def audit_invariants(df: pd.DataFrame) -> Dict[str, Any]:
    """Check core audit invariants specified in docs/audit-history.md."""
    df = normalize_dataframe(df)
    violations = []

    # 1. Invariant: Clean runs must have 0% peer contamination
    clean_runs = df[df["poisoning"] == "clean"]
    clean_contam = clean_runs[clean_runs["contamination_rate"] > 0]
    if not clean_contam.empty:
        violations.append(
            {
                "invariant": "Zero contamination in clean runs",
                "violations_count": len(clean_contam),
                "details": clean_contam[
                    ["environment", "policy", "topology", "seed", "contamination_rate"]
                ].to_dict(orient="records"),
            }
        )

    # 2. Invariant: Off (isolated) topology must have 0% peer contamination even under poisoning
    off_runs = df[df["topology"] == "off"]
    off_contam = off_runs[off_runs["contamination_rate"] > 0]
    if not off_contam.empty:
        violations.append(
            {
                "invariant": "Zero peer contamination with topology=off",
                "violations_count": len(off_contam),
                "details": off_contam[
                    ["environment", "policy", "poisoning", "seed", "contamination_rate"]
                ].to_dict(orient="records"),
            }
        )

    # 3. Check completeness per environment
    completeness = {}
    for env, env_df in df.groupby("environment"):
        completeness[str(env)] = {
            "total_runs": len(env_df),
            "unique_seeds": sorted([int(s) for s in env_df["seed"].unique() if pd.notna(s)]),
            "policies": sorted([str(p) for p in env_df["policy"].unique() if pd.notna(p)]),
            "topologies": sorted([str(t) for t in env_df["topology"].unique() if pd.notna(t)]),
            "poisoning_modes": sorted(
                [str(m) for m in env_df["poisoning"].unique() if pd.notna(m)]
            ),
        }

    return {
        "status": "PASSED" if not violations else "FAILED",
        "violations": violations,
        "completeness": completeness,
    }


def run_statistical_suite(df: pd.DataFrame) -> Tuple[List[HypothesisTestResult], Dict[str, Any]]:
    """Run full hypothesis testing suite on experimental data across environments."""
    df = normalize_dataframe(df)
    audit_results = audit_invariants(df)

    test_specs = [
        # Hyp 1: Structured Incremental vs Naive Overwrite on Reward (Clean, Ring)
        {
            "id": "H1_struct_vs_naive_score",
            "metric": "score",
            "cond_a": {
                "policy": "structured_incremental",
                "topology": "ring",
                "poisoning": "clean",
            },
            "cond_b": {"policy": "naive_overwrite", "topology": "ring", "poisoning": "clean"},
            "label_a": "Structured Incremental (Ring, Clean)",
            "label_b": "Naïve Overwrite (Ring, Clean)",
        },
        # Hyp 2: Structured Incremental vs Naive Overwrite on Reward (Clean, Off)
        {
            "id": "H2_struct_vs_naive_score_isolated",
            "metric": "score",
            "cond_a": {"policy": "structured_incremental", "topology": "off", "poisoning": "clean"},
            "cond_b": {"policy": "naive_overwrite", "topology": "off", "poisoning": "clean"},
            "label_a": "Structured Incremental (Off, Clean)",
            "label_b": "Naïve Overwrite (Off, Clean)",
        },
        # Hyp 3: Structured Incremental vs Raw Trajectory on Reward (Clean, Ring)
        {
            "id": "H3_struct_vs_raw_score",
            "metric": "score",
            "cond_a": {
                "policy": "structured_incremental",
                "topology": "ring",
                "poisoning": "clean",
            },
            "cond_b": {"policy": "raw_trajectory_buffer", "topology": "ring", "poisoning": "clean"},
            "label_a": "Structured Incremental (Ring, Clean)",
            "label_b": "Raw Trajectory Buffer (Ring, Clean)",
        },
        # Hyp 4: Ring vs Full Broadcast on Contamination Rate under internal poisoning
        {
            "id": "H4_ring_vs_broadcast_contamination_rate",
            "metric": "contamination_rate",
            "cond_a": {
                "policy": "structured_incremental",
                "topology": "ring",
                "poisoning": "internal",
            },
            "cond_b": {
                "policy": "structured_incremental",
                "topology": "full_broadcast",
                "poisoning": "internal",
            },
            "label_a": "Ring Topology (Structured, Poisoned)",
            "label_b": "Full Broadcast (Structured, Poisoned)",
        },
        # Hyp 5: Ring vs Full Broadcast on Time-to-Propagation (Latency)
        {
            "id": "H5_ring_vs_broadcast_propagation_latency",
            "metric": "propagation_latency",
            "cond_a": {
                "policy": "structured_incremental",
                "topology": "ring",
                "poisoning": "internal",
            },
            "cond_b": {
                "policy": "structured_incremental",
                "topology": "full_broadcast",
                "poisoning": "internal",
            },
            "label_a": "Ring Topology (Structured, Poisoned)",
            "label_b": "Full Broadcast (Structured, Poisoned)",
        },
        # Hyp 6: Structured Incremental vs Naive Overwrite on Self-BLEU (Collapse)
        {
            "id": "H6_struct_vs_naive_self_bleu",
            "metric": "self_bleu",
            "cond_a": {"policy": "structured_incremental", "topology": "off", "poisoning": "clean"},
            "cond_b": {"policy": "naive_overwrite", "topology": "off", "poisoning": "clean"},
            "label_a": "Structured Incremental (Off, Clean)",
            "label_b": "Naïve Overwrite (Off, Clean)",
        },
        # Hyp 7: Structured Incremental vs Naive Overwrite on Action Entropy
        {
            "id": "H7_struct_vs_naive_action_entropy",
            "metric": "action_entropy",
            "cond_a": {
                "policy": "structured_incremental",
                "topology": "ring",
                "poisoning": "clean",
            },
            "cond_b": {"policy": "naive_overwrite", "topology": "ring", "poisoning": "clean"},
            "label_a": "Structured Incremental (Ring, Clean)",
            "label_b": "Naïve Overwrite (Ring, Clean)",
        },
    ]

    results: List[HypothesisTestResult] = []

    for env in df["environment"].unique():
        env_df = df[df["environment"] == env]

        for spec in test_specs:
            metric = spec["metric"]
            if metric not in env_df.columns:
                continue

            sub_a = env_df.copy()
            for k, v in spec["cond_a"].items():
                sub_a = sub_a[sub_a[k] == v]

            sub_b = env_df.copy()
            for k, v in spec["cond_b"].items():
                sub_b = sub_b[sub_b[k] == v]

            vals_a = sub_a[metric].dropna().values
            vals_b = sub_b[metric].dropna().values

            if len(vals_a) == 0 or len(vals_b) == 0:
                continue

            # Two-sided Mann-Whitney U test
            try:
                stat, p_val = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
            except ValueError:
                stat, p_val = 0.0, 1.0

            delta = compute_cliffs_delta(vals_a, vals_b)
            interp = interpret_cliffs_delta(delta)

            results.append(
                HypothesisTestResult(
                    test_id=spec["id"],
                    environment=str(env),
                    metric=metric,
                    condition_a=spec["label_a"],
                    condition_b=spec["label_b"],
                    n_a=len(vals_a),
                    n_b=len(vals_b),
                    mean_a=float(np.mean(vals_a)),
                    mean_b=float(np.mean(vals_b)),
                    median_a=float(np.median(vals_a)),
                    median_b=float(np.median(vals_b)),
                    u_stat=float(stat),
                    p_value=float(p_val),
                    p_value_fdr=float(p_val),  # Updated below
                    cliffs_delta=float(delta),
                    effect_size_interpretation=interp,
                    is_statistically_significant=bool(p_val < 0.05),
                )
            )

    # Benjamini-Hochberg FDR correction across all tests
    if results:
        raw_p = [r.p_value for r in results]
        n_tests = len(raw_p)
        sorted_indices = np.argsort(raw_p)
        sorted_p = np.array(raw_p)[sorted_indices]

        # FDR computation via Benjamini-Hochberg procedure
        q_vals = np.zeros(n_tests)
        min_q = 1.0
        for i in range(n_tests - 1, -1, -1):
            rank = i + 1
            q = sorted_p[i] * n_tests / rank
            min_q = min(min_q, q)
            q_vals[i] = min_q

        # Map monotonic adjusted q-values back from sorted rank order to the original test order
        fdr_p = np.zeros(n_tests)
        fdr_p[sorted_indices] = np.clip(q_vals, 0.0, 1.0)

        for i, r in enumerate(results):
            r.p_value_fdr = float(fdr_p[i])
            r.is_statistically_significant = bool(r.p_value_fdr < 0.05)

    return results, audit_results
