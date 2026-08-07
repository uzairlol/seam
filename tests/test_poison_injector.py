"""Unit tests for PoisonInjector in seam.poisoning.injector."""

from __future__ import annotations

from seam.memory.naive_overwrite import NaiveOverwritePolicy
from seam.orchestration.config_loader import MemoryConfig, PoisoningConfig, SharingConfig
from seam.poisoning.injector import DEFAULT_POISON_PAYLOADS, PoisonInjector
from seam.sharing.engine import MemorySharingEngine


def test_poison_injector_clean():
    cfg = PoisoningConfig(mode="clean")
    injector = PoisonInjector(cfg, env_type="number_guessing")

    assert not injector.is_active

    policies = {"agent_0": NaiveOverwritePolicy(MemoryConfig(policy="naive_overwrite"))}
    injected = injector.inject_initial_memory(policies)
    assert not injected


def test_poison_injector_internal():
    cfg = PoisoningConfig(mode="internal", poison_agent_id="agent_0")
    injector = PoisonInjector(cfg, env_type="number_guessing")

    assert injector.is_active
    assert "guess '100'" in injector.poison_payload

    policy = NaiveOverwritePolicy(MemoryConfig(policy="naive_overwrite"))
    policies = {"agent_0": policy}

    injected = injector.inject_initial_memory(policies)
    assert injected
    assert "guess '100'" in policy.get_context()


def test_poison_injector_channel():
    cfg = PoisoningConfig(mode="channel", poison_agent_id="agent_0")
    injector = PoisonInjector(cfg, env_type="bargaining_game")

    sharing_cfg = SharingConfig(mode="broadcast", publish_every_n_rounds=1)
    engine = MemorySharingEngine(config=sharing_cfg, n_agents=2)

    injected = injector.inject_channel(engine, round_num=1)
    assert injected
    assert "0 100" in engine.get_shared_context("agent_0")
