"""Efficacy-gap normalization relative to ``no_memory`` and oracle references.

The raw ``final_score`` saturates and is hard to compare across conditions (a
perfect score is rare; scores hover near the ``no_memory`` floor).  The
efficacy gap rescales each run onto a per-seed headroom:

    efficacy_gap = (score - baseline) / (oracle - baseline)

where *baseline* is the mean ``no_memory`` score for the same seed and *oracle*
is the deterministic optimal score for that (env_type, seed) instance.  The
rescaled value is comparable across environments and expresses how much of the
recoverable performance gap each condition actually recovered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_CONTROL_POLICY = "no_memory"
_CONTROL_TOPOLOGY = "off"

_REQUIRED = ("policy", "topology", "seed", "final_score", "oracle_score")


def compute_efficacy_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Add an ``efficacy_gap`` column to *df*, normalized per seed.

    Runs whose seed has no ``no_memory``/``off`` baseline or no oracle score get
    a NaN efficacy gap and are excluded from downstream aggregation.

    Args:
        df: DataFrame with ``policy``, ``topology``, ``seed``, ``final_score``
            and ``oracle_score`` columns.

    Returns:
        A copy of *df* with the extra ``efficacy_gap`` column.
    """
    result = df.copy()
    if result.empty or any(col not in result.columns for col in _REQUIRED):
        return result

    baseline_map, oracle_map = _control_stats(result)
    if not baseline_map or not oracle_map:
        result["efficacy_gap"] = np.nan
        return result

    baseline = result["seed"].map(baseline_map)
    oracle = result["seed"].map(oracle_map)
    headroom = oracle - baseline

    used = headroom.abs() > 1e-9
    result["efficacy_gap"] = np.nan
    result.loc[used, "efficacy_gap"] = (
        result.loc[used, "final_score"] - baseline.loc[used]
    ) / headroom.loc[used]
    return result


def _control_stats(df: pd.DataFrame) -> tuple[dict[object, float], dict[object, float]]:
    """Return ``{seed: baseline}`` and ``{seed: oracle}`` reference mappings."""
    baseline_rows = df[(df["policy"] == _CONTROL_POLICY) & (df["topology"] == _CONTROL_TOPOLOGY)]
    if baseline_rows.empty:
        baseline_rows = df[df["policy"] == _CONTROL_POLICY]
    if baseline_rows.empty:
        return {}, {}

    baseline_map = baseline_rows.groupby("seed")["final_score"].mean().to_dict()
    oracle_map = df.groupby("seed")["oracle_score"].mean().to_dict()
    return baseline_map, oracle_map
