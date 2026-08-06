"""Unit tests for RawTrajectoryBufferPolicy sliding window and serialization."""

from __future__ import annotations

from seam.memory.raw_trajectory import RawTrajectoryBufferPolicy


def test_raw_trajectory_buffer_sliding_window() -> None:
    """Test buffer retains at most window_size entries."""
    policy = RawTrajectoryBufferPolicy(window_size=3)
    assert policy.get_context() == ""

    for i in range(5):
        policy.update({
            "observation": {"step": i},
            "action": f"move_{i}",
            "reward": float(i),
        })

    ctx = policy.get_context()
    assert "Step 1: Obs={'step': 2}" in ctx
    assert "Step 3: Obs={'step': 4}" in ctx
    assert "step': 0" not in ctx  # older entries evicted


def test_raw_trajectory_buffer_serialization() -> None:
    """Test to_dict and from_dict serialization."""
    policy = RawTrajectoryBufferPolicy(window_size=5)
    policy.update({"observation": {"pos": [0, 0]}, "action": "harvest", "reward": 1.0})

    data = policy.to_dict()
    assert data["policy"] == "raw_trajectory_buffer"
    assert data["window_size"] == 5
    assert len(data["buffer"]) == 1

    new_policy = RawTrajectoryBufferPolicy()
    new_policy.from_dict(data)
    assert new_policy.window_size == 5
    assert "harvest" in new_policy.get_context()
