"""Unit tests for the inferential statistics suite (Mann-Whitney U, Cliff's delta, FDR, audit)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from seam.analysis.significance import (
    audit_invariants,
    compute_cliffs_delta,
    interpret_cliffs_delta,
    run_statistical_suite,
)

_POLICIES = ["naive_overwrite", "raw_trajectory_buffer", "structured_incremental"]
_TOPOLOGIES = ["off", "ring", "full_broadcast"]
_POISONING = ["clean", "internal"]
_ENVIRONMENTS = ["resource_foraging", "number_guessing"]
_SEEDS = list(range(40, 50))
_N = len(_SEEDS)


def _build_synthetic_df() -> pd.DataFrame:
    """Return a compact synthetic experiment dataframe with known effect directions."""
    rng = np.random.default_rng(0)
    rows: list[dict[str, float | str]] = []
    for env in _ENVIRONMENTS:
        for policy in _POLICIES:
            for topology in _TOPOLOGIES:
                for poisoning in _POISONING:
                    for seed in _SEEDS:
                        # Structured curation outperforms naive/raw on task score under
                        # clean conditions (the direction claimed by the manuscript).
                        if policy == "structured_incremental":
                            score = 0.12 + rng.normal(0, 0.02)
                        else:
                            score = 0.01 + rng.normal(0, 0.015)

                        # Ring propagates slower than full broadcast; off never propagates.
                        if poisoning == "internal" and topology == "full_broadcast":
                            latency = 1.0
                        elif poisoning == "internal" and topology == "ring":
                            latency = 3.5 + rng.normal(0, 0.3)
                        else:
                            latency = np.nan
                        contamination = (
                            1.0 if poisoning == "internal" and topology != "off" else 0.0
                        )

                        self_bleu = {
                            ("structured_incremental", "off"): 0.995,
                            ("naive_overwrite", "off"): 0.999,
                            ("raw_trajectory_buffer", "off"): 0.998,
                        }.get((policy, topology), 0.99) + rng.normal(0, 0.0005)

                        rows.append(
                            {
                                "environment": env,
                                "policy": policy,
                                "topology": topology,
                                "poisoning": poisoning,
                                "seed": seed,
                                "score": max(0.0, score),
                                "self_bleu": self_bleu,
                                "contamination_rate": contamination,
                                "propagation_latency": latency,
                            }
                        )
    return pd.DataFrame(rows)


def test_compute_cliffs_delta() -> None:
    """Cliff's delta of a fully-separated pair is +1.0/-1.0; identical pairs are 0."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    assert compute_cliffs_delta(x, y) == -1.0
    assert compute_cliffs_delta(y, x) == 1.0
    assert compute_cliffs_delta(x, x) == 0.0


def test_interpret_cliffs_delta_bands() -> None:
    """Standard interpretation bands for Cliff's delta magnitude."""
    assert interpret_cliffs_delta(0.05) == "negligible"
    assert interpret_cliffs_delta(0.2) == "small"
    assert interpret_cliffs_delta(0.4) == "medium"
    assert interpret_cliffs_delta(0.9) == "large"


def test_run_statistical_suite_hypotheses() -> None:
    """The suite emits every defined hypothesis, including H5 (latency)."""
    df = _build_synthetic_df()
    results, audit = run_statistical_suite(df)

    by_id: dict[str, list[object]] = {}
    for r in results:
        by_id.setdefault(r.test_id, []).append(r)

    # Every hypothesis ID in the suite was generated.
    assert (
        "H1_struct_vs_naive_score" in by_id
        and "H2_struct_vs_naive_score_isolated" in by_id
        and "H3_struct_vs_raw_score" in by_id
        and "H4_ring_vs_broadcast_contamination_rate" in by_id
        and "H5_ring_vs_broadcast_propagation_latency" in by_id
        and "H6_struct_vs_naive_self_bleu" in by_id
    )

    # H5 must compare ring vs broadcast latency for the poisoned structured condition.
    h5 = by_id["H5_ring_vs_broadcast_propagation_latency"][0]
    assert h5.metric == "propagation_latency"
    assert h5.n_a == _N
    assert h5.n_b == _N
    assert h5.mean_a > h5.mean_b  # ring (3.5) slower than broadcast (1.0)
    assert h5.is_statistically_significant

    # H1 should reproduce the structured > naive score advantage.
    h1 = by_id["H1_struct_vs_naive_score"][0]
    assert h1.mean_a > h1.mean_b
    assert h1.is_statistically_significant

    # H4: contamination rate is 100% for both topologies -> no difference, not significant.
    h4 = by_id["H4_ring_vs_broadcast_contamination_rate"][0]
    assert h4.mean_a == 1.0 and h4.mean_b == 1.0
    assert not h4.is_statistically_significant

    assert audit["status"] == "PASSED"


def test_audit_invariants_detect_clean_contamination() -> None:
    """Invariant audit flags contamination in clean conditions."""
    df = pd.DataFrame(
        [
            {
                "environment": "resource_foraging",
                "policy": "naive_overwrite",
                "topology": "off",
                "poisoning": "clean",
                "seed": 40,
                "score": 0.2,
                "self_bleu": 0.9,
                "contamination_rate": 0.5,  # should never happen in a clean run
            }
        ]
    )
    audit = audit_invariants(df)
    assert audit["status"] == "FAILED"
    assert any(v["invariant"] == "Zero contamination in clean runs" for v in audit["violations"])


def test_audit_invariants_detect_off_topology_contamination() -> None:
    """Invariant audit flags peer contamination when sharing is off."""
    df = pd.DataFrame(
        [
            {
                "environment": "resource_foraging",
                "policy": "naive_overwrite",
                "topology": "off",
                "poisoning": "internal",
                "seed": 40,
                "score": 0.2,
                "self_bleu": 0.9,
                "contamination_rate": 1.0,  # impossible with topology off
            }
        ]
    )
    audit = audit_invariants(df)
    assert audit["status"] == "FAILED"
    assert any(
        v["invariant"] == "Zero peer contamination with topology=off" for v in audit["violations"]
    )
