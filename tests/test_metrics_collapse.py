"""Unit tests for memory collapse metrics (Self-BLEU, cosine similarity, action entropy, length)."""

from __future__ import annotations

from unittest.mock import MagicMock

from seam.metrics.collapse import (
    compute_action_entropy,
    compute_cosine_similarity,
    compute_embedding_similarity,
    compute_memory_length,
    compute_self_bleu,
)


def test_compute_self_bleu_identical_sequences() -> None:
    """Identical memory states yield high Self-BLEU score near 1.0."""
    memories = [
        "Harvest food at position 2 2",
        "Harvest food at position 2 2",
        "Harvest food at position 2 2",
    ]
    score = compute_self_bleu(memories)
    assert 0.8 <= score <= 1.0


def test_compute_self_bleu_diverse_sequences() -> None:
    """Diverse memory states yield lower Self-BLEU score."""
    memories = [
        "Harvest food at position 2 2",
        "Avoid water hazard on left boundary",
        "Negotiate 50 50 split with peer agent",
    ]
    score = compute_self_bleu(memories)
    assert score < 0.5


def test_compute_cosine_similarity() -> None:
    """Test vector cosine similarity math."""
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]

    assert abs(compute_cosine_similarity(v1, v2) - 1.0) < 1e-5
    assert abs(compute_cosine_similarity(v1, v3) - 0.0) < 1e-5


def test_compute_embedding_similarity_fallback() -> None:
    """Test embedding similarity fallback without client."""
    memories = ["Harvest food", "Harvest food", "Explore top right"]
    sims = compute_embedding_similarity(memories, client=None)
    assert len(sims) == 2
    assert sims[0] == 1.0  # Identical token overlap
    assert sims[1] < 1.0   # Different token overlap


def test_compute_action_entropy() -> None:
    """Test rolling action entropy calculation."""
    repetitive_actions = ["harvest"] * 10
    entropies = compute_action_entropy(repetitive_actions, window_size=5)
    assert len(entropies) == 6
    assert all(e == 0.0 for e in entropies)

    diverse_actions = ["up", "down", "left", "right", "harvest"] * 2
    div_entropies = compute_action_entropy(diverse_actions, window_size=5)
    assert div_entropies[0] > 1.5


def test_compute_memory_length() -> None:
    """Test memory word length calculation."""
    memories = ["Short rule", "A slightly longer memory rule string for testing"]
    lengths = compute_memory_length(memories)
    assert lengths == [2, 8]
