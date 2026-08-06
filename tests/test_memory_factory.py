"""Unit tests for create_memory_policy factory."""

from __future__ import annotations

import pytest

from seam.memory import (
    NaiveOverwritePolicy,
    RawTrajectoryBufferPolicy,
    StructuredIncrementalPolicy,
    create_memory_policy,
)
from seam.orchestration.config_loader import MemoryConfig


def test_create_memory_policy_naive_overwrite() -> None:
    """Factory should instantiate NaiveOverwritePolicy."""
    cfg = MemoryConfig(policy="naive_overwrite", max_tokens=128)
    pol = create_memory_policy(cfg)
    assert isinstance(pol, NaiveOverwritePolicy)
    assert pol.max_tokens == 128


def test_create_memory_policy_raw_trajectory() -> None:
    """Factory should instantiate RawTrajectoryBufferPolicy."""
    cfg = MemoryConfig(policy="raw_trajectory_buffer", window_size=7)
    pol = create_memory_policy(cfg)
    assert isinstance(pol, RawTrajectoryBufferPolicy)
    assert pol.window_size == 7


def test_create_memory_policy_structured() -> None:
    """Factory should instantiate StructuredIncrementalPolicy."""
    cfg = MemoryConfig(policy="structured_incremental", max_playbook_entries=12)
    pol = create_memory_policy(cfg)
    assert isinstance(pol, StructuredIncrementalPolicy)
    assert pol.max_playbook_entries == 12


def test_create_memory_policy_invalid() -> None:
    """Factory should raise ValueError for invalid policy name."""
    cfg = MemoryConfig(policy="nonexistent_policy")
    with pytest.raises(ValueError) as exc_info:
        create_memory_policy(cfg)
    assert "Unknown memory policy" in str(exc_info.value)
