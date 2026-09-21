"""Time-to-infection survival analysis for peer contamination.

Each run records the first round at which every peer agent's memory tested
positive for the poison payload (``peer_propagation_round``; ``None`` for peers
that stayed clean).  Peers that never become contaminated are right-censored at
their episode length.  This module builds per-condition *survival curves* — the
fraction of not-yet-contaminated peers still at risk as a function of round —
estimating a Kaplan–Meier survival function from event/censor times.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

_GROUP_COLS = ("policy", "topology", "poisoning_mode")


def build_survival_dataframe(summaries: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Flatten run summaries into per-peer event/censor records.

    Args:
        summaries: Iterable of run summary dicts containing
            ``peer_propagation_round``, ``rounds_played`` and optional group
            keys (``policy``, ``topology``, ``poisoning_mode``).

    Returns:
        Long DataFrame with columns ``run_id``, ``peer``, ``event_round``
        (NaN = censored), ``censored``, ``episode_length`` and any group keys.
    """
    rows: list[dict[str, Any]] = []
    for index, summary in enumerate(summaries):
        rounds_map = summary.get("peer_propagation_round") or {}
        episode_length = int(summary.get("rounds_played", 0))
        group = {key: summary.get(key) for key in _GROUP_COLS if key in summary}
        run_id = summary.get("run_id", index)
        for peer, first_round in rounds_map.items():
            censored = first_round is None
            rows.append(
                {
                    **group,
                    "run_id": run_id,
                    "peer": peer,
                    "event_round": (float(first_round) if not censored else np.nan),
                    "censored": censored,
                    "episode_length": episode_length,
                }
            )
    return pd.DataFrame(rows)


def compute_survival_curves(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate Kaplan–Meier survival curves from per-peer event/censor records.

    Args:
        df: Output of :func:`build_survival_dataframe`.

    Returns:
        Long DataFrame with ``round``, ``at_risk``, ``infected``,
        ``survival_prob`` columns plus any group keys.
    """
    if df.empty or "event_round" not in df.columns:
        return pd.DataFrame()

    group_cols = [c for c in _GROUP_COLS if c in df.columns]
    horizon = int(max(df["event_round"].fillna(df["episode_length"]).max(), 1))

    frames: list[pd.DataFrame] = []
    keys = df.groupby(group_cols, dropna=False).groups if group_cols else {"all": df.index}
    for key_values, indices in keys.items():
        subset = df.loc[indices].copy()
        group_row: dict[str, Any] = {}
        if group_cols:
            group_row = dict(
                zip(
                    group_cols,
                    key_values if isinstance(key_values, tuple) else (key_values,),
                    strict=True,
                )
            )

        rows: list[dict[str, Any]] = []
        survival = 1.0
        for t in range(1, horizon + 1):
            at_risk = _at_risk_count(subset, t)
            infected = int(np.sum(subset["event_round"] == t))
            if at_risk > 0:
                survival *= 1.0 - infected / at_risk
            rows.append(
                {
                    **group_row,
                    "round": t,
                    "at_risk": at_risk,
                    "infected": infected,
                    "survival_prob": float(survival),
                    "n_peers": int(len(subset)),
                }
            )
        frames.append(pd.DataFrame(rows))

    return pd.concat(frames, ignore_index=True)


def mean_time_to_infection(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize the mean (first-contaminated) round per group.

    Note: this ignores censoring to give a quick interpretable number; the
    survival curve itself is the censoring-aware estimate.

    Args:
        df: Output of :func:`build_survival_dataframe`.

    Returns:
        Grouped DataFrame with ``mean_time_to_infection`` and ``infected_fraction``.
    """
    if df.empty:
        return pd.DataFrame()

    group_cols = [c for c in _GROUP_COLS if c in df.columns]
    summary = df.groupby(group_cols, dropna=False).agg(
        mean_time_to_infection=(pd.NamedAgg(column="event_round", aggfunc="mean")),
        infected_fraction=(pd.NamedAgg(column="censored", aggfunc=lambda s: 1.0 - s.mean())),
        n_peers=(pd.NamedAgg(column="peer", aggfunc="count")),
    )
    return summary.reset_index()


def _at_risk_count(df: pd.DataFrame, t: int) -> int:
    """Number of peers neither infected before round *t* nor censored before it."""
    not_yet_infected = df["event_round"].isna() | (df["event_round"] >= t)
    alive_until_t = df["episode_length"] >= t
    return int(np.sum(not_yet_infected & alive_until_t))
