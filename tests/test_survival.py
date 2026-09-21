"""Unit tests for time-to-infection survival analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seam.analysis.survival import (
    build_survival_dataframe,
    compute_survival_curves,
    mean_time_to_infection,
)


def _summaries() -> list[dict]:
    return [
        {
            "run_id": "r1",
            "policy": "no_memory",
            "topology": "ring",
            "poisoning_mode": "internal",
            "rounds_played": 10,
            "peer_propagation_round": {
                "agent_1": 3,
                "agent_2": None,
                "agent_3": 5,
            },
        },
        {
            "run_id": "r2",
            "policy": "no_memory",
            "topology": "ring",
            "poisoning_mode": "internal",
            "rounds_played": 10,
            "peer_propagation_round": {
                "agent_1": 4,
                "agent_2": 6,
                "agent_3": None,
            },
        },
    ]


def test_build_survival_dataframe_long_format() -> None:
    df = build_survival_dataframe(_summaries())
    assert len(df) == 6
    assert df["censored"].sum() == 2
    assert set(df.columns) >= {
        "run_id",
        "peer",
        "event_round",
        "censored",
        "episode_length",
        "policy",
        "topology",
        "poisoning_mode",
    }
    censored = df[df["censored"]]
    assert censored["event_round"].isna().all()


def test_compute_survival_curves_monotone_decreasing() -> None:
    df = build_survival_dataframe(_summaries())
    curves = compute_survival_curves(df)
    assert "survival_prob" in curves.columns
    assert "round" in curves.columns
    # Survival must be non-increasing in round.
    for _, group in curves.groupby(["policy", "topology", "poisoning_mode"]):
        probs = group["survival_prob"].to_numpy()
        assert np.all(np.diff(probs) <= 1e-12)
    # After the first event, survival < 1.
    assert (curves["survival_prob"] < 1.0).any()
    # Peak survival is 1.0 at round 1 (nothing infected yet).
    assert (curves.loc[curves["round"] == 1, "survival_prob"] == 1.0).all()


def test_mean_time_to_infection() -> None:
    df = build_survival_dataframe(_summaries())
    summary = mean_time_to_infection(df)
    assert len(summary) == 1
    # Infected peers: 3, 5, 4, 6 -> mean = 4.5
    assert pytest.approx(summary.iloc[0]["mean_time_to_infection"]) == 4.5
    assert pytest.approx(summary.iloc[0]["infected_fraction"]) == 4 / 6


def test_compute_survival_curves_empty() -> None:
    assert compute_survival_curves(pd.DataFrame()).empty
