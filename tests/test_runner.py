"""Unit tests for EpisodeRunner execution loop, scoring, and metrics aggregation."""

from __future__ import annotations

from pathlib import Path
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


def test_episode_runner_execution_and_summary(tmp_path: Path) -> None:
    """EpisodeRunner executes a full episode loop and outputs evaluation summary."""
    cfg = ExperimentConfig(
        experiment_id="test_runner_exp",
        description="EpisodeRunner unit test",
        env=EnvConfig(type="resource_foraging", n_agents=2, episode_length=5),
        model=ModelConfig(model_name="qwen2.5:7b-instruct", base_url="http://localhost:11434"),
        memory=MemoryConfig(policy="naive_overwrite"),
        sharing=SharingConfig(mode="off"),
        poisoning=PoisoningConfig(mode="clean"),
        seeds=[42],
    )

    mock_client = MagicMock()
    mock_client.complete.return_value = ("Action: harvest", 20)

    runner = EpisodeRunner(config=cfg, seed=42, client=mock_client, base_dir=tmp_path)
    summary = runner.run()

    assert summary["seed"] == 42
    assert summary["rounds_played"] == 5
    assert 0.0 <= summary["final_score"] <= 1.0
    assert "mean_self_bleu" in summary
    assert len(summary["cumulative_rewards"]) == 2
    assert (tmp_path / summary["run_id"] / "events.jsonl").exists()
    assert (tmp_path / summary["run_id"] / "summary.json").exists()
