"""Unit tests for efficacy-gap normalization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seam.analysis.efficacy import compute_efficacy_gap


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Baseline: no_memory / off, seeds 42 & 43
            {
                "policy": "no_memory",
                "topology": "off",
                "poisoning_mode": "clean",
                "seed": 42,
                "final_score": 0.1,
                "oracle_score": 0.5,
            },
            {
                "policy": "no_memory",
                "topology": "off",
                "poisoning_mode": "clean",
                "seed": 43,
                "final_score": 0.2,
                "oracle_score": 0.6,
            },
            # Treated conditions, same seeds
            {
                "policy": "naive_overwrite",
                "topology": "ring",
                "poisoning_mode": "clean",
                "seed": 42,
                "final_score": 0.3,
                "oracle_score": 0.5,
            },
            {
                "policy": "structured_incremental",
                "topology": "ring",
                "poisoning_mode": "internal",
                "seed": 43,
                "final_score": 0.4,
                "oracle_score": 0.6,
            },
        ]
    )


def test_efficacy_gap_normalizes_to_headroom() -> None:
    df = compute_efficacy_gap(_df())
    # seed 42: baseline=0.1, oracle=0.5 -> headroom 0.4; naive at 0.3 -> 0.5
    assert pytest.approx(df.loc[2, "efficacy_gap"], abs=1e-9) == 0.5
    # baseline policy gets efficacy 0
    assert pytest.approx(df.loc[0, "efficacy_gap"], abs=1e-9) == 0.0
    # seed 43: baseline=0.2, oracle=0.6 -> headroom 0.4; structured at 0.4 -> 0.5
    assert pytest.approx(df.loc[3, "efficacy_gap"], abs=1e-9) == 0.5


def test_efficacy_gap_unchanged_without_oracle_column() -> None:
    df = _df().drop(columns=["oracle_score"])
    out = compute_efficacy_gap(df)
    assert "efficacy_gap" not in out.columns  # cannot compute without an oracle reference
    assert out["final_score"].equals(df["final_score"])


def test_efficacy_gap_empty_frame() -> None:
    assert compute_efficacy_gap(pd.DataFrame()).empty


def test_efficacy_gap_returns_nan_when_no_baseline() -> None:
    df = _df()
    df = df[df["policy"] != "no_memory"].copy()
    out = compute_efficacy_gap(df)
    assert out["efficacy_gap"].isna().all()


def test_efficacy_gap_rolls_back_to_any_no_memory_row() -> None:
    # No topology=off baseline exists; fall back to all no_memory rows for the seed.
    df = _df()
    df.loc[0, "topology"] = "ring"
    df.loc[1, "topology"] = "ring"
    out = compute_efficacy_gap(df)
    assert np.isfinite(out["efficacy_gap"].iloc[0])
