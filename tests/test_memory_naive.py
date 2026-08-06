"""Unit tests for NaiveOverwritePolicy serialization and update logic."""

from __future__ import annotations

from unittest.mock import MagicMock

from seam.memory.naive_overwrite import NaiveOverwritePolicy


def test_naive_overwrite_initialization_and_reset() -> None:
    """Test memory policy reset and context retrieval."""
    policy = NaiveOverwritePolicy(max_tokens=256, initial_memory="Initial rule")
    assert policy.get_context() == "Initial rule"

    policy.reset()
    assert policy.get_context() == ""


def test_naive_overwrite_update_with_mock_client() -> None:
    """Test memory update with mocked OllamaClient."""
    mock_client = MagicMock()
    mock_client.complete.return_value = ("New updated reflection memory", 30)

    policy = NaiveOverwritePolicy()
    experience = {"observation": {"pos": [1, 1]}, "action": "harvest", "reward": 1.0}
    updated = policy.update(experience, client=mock_client)

    assert updated == "New updated reflection memory"
    assert policy.get_context() == "New updated reflection memory"
    mock_client.complete.assert_called_once()


def test_naive_overwrite_update_deterministic_fallback() -> None:
    """Test memory update fallback when client is None."""
    policy = NaiveOverwritePolicy()
    experience = {"observation": {"pos": [1, 1]}, "action": "harvest", "reward": 1.0}
    updated = policy.update(experience, client=None)

    assert updated == "Last Action: harvest | Reward: 1.0"


def test_naive_overwrite_serialization() -> None:
    """Test to_dict and from_dict serialization cycle."""
    policy = NaiveOverwritePolicy(max_tokens=512, initial_memory="Saved strategy")
    data = policy.to_dict()

    assert data["policy"] == "naive_overwrite"
    assert data["max_tokens"] == 512
    assert data["memory_text"] == "Saved strategy"

    new_policy = NaiveOverwritePolicy()
    new_policy.from_dict(data)

    assert new_policy.max_tokens == 512
    assert new_policy.get_context() == "Saved strategy"
