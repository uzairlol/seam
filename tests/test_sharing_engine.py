"""Unit tests for MemorySharingEngine in seam.sharing.engine."""

from __future__ import annotations

import pytest

from seam.memory.naive_overwrite import NaiveOverwritePolicy
from seam.orchestration.config_loader import MemoryConfig, SharingConfig
from seam.sharing.engine import MemorySharingEngine


def create_dummy_policies(n_agents: int) -> dict[str, NaiveOverwritePolicy]:
    mem_cfg = MemoryConfig(policy="naive_overwrite")
    policies = {}
    for i in range(n_agents):
        aid = f"agent_{i}"
        policy = NaiveOverwritePolicy(mem_cfg)
        policy.update({"action": f"act_{i}", "reward": 1.0, "observation": {}}, client=None)
        policies[aid] = policy
    return policies


def test_sharing_engine_off():
    cfg = SharingConfig(mode="off", publish_every_n_rounds=1)
    engine = MemorySharingEngine(config=cfg, n_agents=4)

    policies = create_dummy_policies(4)
    counts = engine.step(round_num=1, memory_policies=policies)

    assert not engine.is_active
    assert sum(counts.values()) == 0
    assert engine.get_shared_context("agent_0") == ""


def test_sharing_engine_broadcast():
    cfg = SharingConfig(mode="broadcast", publish_every_n_rounds=1)
    engine = MemorySharingEngine(config=cfg, n_agents=4, topology_type="full_broadcast")

    policies = create_dummy_policies(4)
    counts = engine.step(round_num=1, memory_policies=policies)

    assert engine.is_active
    # Every agent receives from 3 peers
    assert counts["agent_0"] == 3
    ctx = engine.get_shared_context("agent_0")
    assert "=== Shared Peer Memories ===" in ctx
    assert "[agent_1]" in ctx
    assert "[agent_2]" in ctx
    assert "[agent_3]" in ctx


test_sharing_engine_cadence_data = [
    (1, 0),  # Round 1 (not publish round) -> 0 messages
    (2, 3),  # Round 2 (publish round) -> 3 messages
]


@pytest.mark.parametrize("round_num, expected_count", test_sharing_engine_cadence_data)
def test_sharing_engine_cadence(round_num: int, expected_count: int):
    cfg = SharingConfig(mode="broadcast", publish_every_n_rounds=2)
    engine = MemorySharingEngine(config=cfg, n_agents=4, topology_type="full_broadcast")

    policies = create_dummy_policies(4)
    counts = engine.step(round_num=round_num, memory_policies=policies)

    assert counts["agent_0"] == expected_count


def test_sharing_engine_ring_topology():
    cfg = SharingConfig(mode="broadcast", publish_every_n_rounds=1)
    engine = MemorySharingEngine(config=cfg, n_agents=4, topology_type="ring")

    policies = create_dummy_policies(4)
    counts = engine.step(round_num=1, memory_policies=policies)

    # In ring with 4 agents, agent_0 receives from agent_3 and agent_1
    assert counts["agent_0"] == 2
    ctx = engine.get_shared_context("agent_0")
    assert "[agent_1]" in ctx
    assert "[agent_3]" in ctx
    assert "[agent_2]" not in ctx
