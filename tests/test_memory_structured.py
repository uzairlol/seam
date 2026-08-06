"""Unit tests for StructuredIncrementalPolicy ACE playbook updates."""

from __future__ import annotations

from unittest.mock import MagicMock

from seam.memory.structured_incremental import StructuredIncrementalPolicy


def test_structured_incremental_llm_reflection() -> None:
    """Test ADD and DEPRECATE instructions parsing from mock LLM response."""
    mock_client = MagicMock()
    mock_client.complete.return_value = (
        "ADD: Always check adjacent cells before harvesting.\n"
        "DEPRECATE: Rule #1",
        40,
    )

    policy = StructuredIncrementalPolicy(max_playbook_entries=10)
    # Seed initial rule
    policy._add_rule("Old initial rule")

    policy.update(
        {"observation": {"pos": [2, 2]}, "action": "harvest", "reward": 1.0},
        client=mock_client,
    )

    ctx = policy.get_context()
    assert "Rule #2: Always check adjacent cells" in ctx
    assert "Rule #1:" not in ctx  # Rule #1 was deprecated


def test_structured_incremental_deterministic_fallback() -> None:
    """Test rule addition when client is None."""
    policy = StructuredIncrementalPolicy(max_playbook_entries=5)
    policy.update({"observation": {}, "action": "move_up", "reward": 0.5}, client=None)

    ctx = policy.get_context()
    assert "Rule #1: Action 'move_up' yielded positive reward 0.50" in ctx


def test_structured_incremental_pruning() -> None:
    """Test max_playbook_entries pruning evicts excess active rules."""
    policy = StructuredIncrementalPolicy(max_playbook_entries=2)
    for i in range(4):
        policy._add_rule(f"Strategy {i}")

    policy._prune_playbook()
    active_count = len([e for e in policy._playbook if e["status"] == "active"])
    assert active_count == 2
    ctx = policy.get_context()
    assert "Strategy 2" in ctx
    assert "Strategy 3" in ctx


def test_structured_incremental_serialization() -> None:
    """Test serialization and deserialization of playbook state."""
    policy = StructuredIncrementalPolicy(max_playbook_entries=15)
    policy._add_rule("Rule Alpha")
    data = policy.to_dict()

    assert data["policy"] == "structured_incremental"
    assert data["max_playbook_entries"] == 15

    new_policy = StructuredIncrementalPolicy()
    new_policy.from_dict(data)
    assert "Rule Alpha" in new_policy.get_context()
