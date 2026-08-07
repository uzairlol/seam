"""Integration unit test verifying MemorySharingEngine inside EpisodeRunner."""

from __future__ import annotations

from unittest.mock import MagicMock

from seam.orchestration.config_loader import (
    EnvConfig,
    ExperimentConfig,
    MemoryConfig,
    ModelConfig,
    PoisoningConfig,
    SharingConfig,
)
from seam.orchestration.runner import EpisodeRunner


def test_sharing_integration_with_runner():
    cfg = ExperimentConfig(
        experiment_id="test_sharing_integration",
        description="Verify multi-agent memory sharing propagation in EpisodeRunner",
        env=EnvConfig(type="number_guessing", n_agents=4, episode_length=4),
        model=ModelConfig(model_name="dummy_model", base_url="http://localhost:11434"),
        memory=MemoryConfig(policy="naive_overwrite"),
        sharing=SharingConfig(mode="broadcast", topology="full_broadcast", publish_every_n_rounds=1),
        poisoning=PoisoningConfig(mode="clean"),
        seeds=[42],
    )

    mock_client = MagicMock()
    mock_client.complete.return_value = ("50", 10)

    with EpisodeRunner(config=cfg, seed=42, client=mock_client) as runner:
        summary = runner.run()

    assert summary["rounds_played"] == 4
    assert summary["final_score"] >= 0.0

    # Collect prompts sent to complete()
    prompts = [call.args[0] for call in mock_client.complete.call_args_list]
    # 4 agents * 4 rounds * 2 calls per step (1 act call + 1 memory update call) = 32 calls
    assert len(prompts) == 32

    # In round 2+, shared memories should appear in agent action prompts
    shared_memory_prompts = [p for p in prompts if "=== Shared Peer Memories ===" in p]
    assert len(shared_memory_prompts) > 0
