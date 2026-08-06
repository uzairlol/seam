"""Factory function for creating memory policies based on configuration."""

from __future__ import annotations

from seam.memory.base_memory import BaseMemoryPolicy
from seam.memory.naive_overwrite import NaiveOverwritePolicy
from seam.memory.raw_trajectory import RawTrajectoryBufferPolicy
from seam.memory.structured_incremental import StructuredIncrementalPolicy
from seam.orchestration.config_loader import MemoryConfig


def create_memory_policy(config: MemoryConfig) -> BaseMemoryPolicy:
    """Instantiate a memory policy instance from a :class:`MemoryConfig`.

    Args:
        config: Memory configuration defining the policy type and parameters.

    Returns:
        An instance of :class:`BaseMemoryPolicy`.

    Raises:
        ValueError: If an unknown policy type is specified.
    """
    policy_type = config.policy.lower().strip()

    if policy_type == "naive_overwrite":
        return NaiveOverwritePolicy(max_tokens=config.max_tokens)
    elif policy_type == "raw_trajectory_buffer":
        return RawTrajectoryBufferPolicy(window_size=config.window_size)
    elif policy_type == "structured_incremental":
        return StructuredIncrementalPolicy(max_playbook_entries=config.max_playbook_entries)
    else:
        raise ValueError(
            f"Unknown memory policy '{config.policy}'. "
            "Supported policies: 'naive_overwrite', 'raw_trajectory_buffer', 'structured_incremental'."
        )
