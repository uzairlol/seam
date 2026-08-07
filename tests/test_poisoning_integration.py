"""Integration unit test verifying PoisonInjector inside EpisodeRunner."""

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


def test_poisoning_integration_with_runner():
    cfg = ExperimentConfig(
        experiment_id="test_poisoning_integration",
        description="Verify memory poisoning condition and contamination metrics in EpisodeRunner",
        env=EnvConfig(type="number_guessing", n_agents=4, episode_length=3),
        model=ModelConfig(model_name="dummy_model", base_url="http://localhost:11434"),
        memory=MemoryConfig(policy="naive_overwrite"),
        sharing=SharingConfig(mode="broadcast", topology="full_broadcast", publish_every_n_rounds=1),
        poisoning=PoisoningConfig(mode="internal", poison_agent_id="agent_0"),
        seeds=[42],
    )

    mock_client = MagicMock()
    mock_client.complete.return_value = ("100", 10)

    with EpisodeRunner(config=cfg, seed=42, client=mock_client) as runner:
        summary = runner.run()

    assert summary["rounds_played"] == 3
    assert "peer_contamination_rate" in summary
    assert "per_agent_poison_adherence" in summary
    assert "agent_0" in summary["per_agent_poison_adherence"]
