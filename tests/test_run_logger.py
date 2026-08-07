"""Unit tests for RunLogger directory initialization, JSONL logging, and summary writing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from seam.logging.run_logger import RunLogger
from seam.orchestration.config_loader import (
    EnvConfig,
    ExperimentConfig,
    MemoryConfig,
    ModelConfig,
    PoisoningConfig,
    SharingConfig,
)


@pytest.fixture
def experiment_config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="test_logging_exp",
        description="RunLogger unit test",
        env=EnvConfig(type="resource_foraging"),
        model=ModelConfig(
            model_name="qwen2.5:7b-instruct",
            base_url="http://localhost:11434",
        ),
        memory=MemoryConfig(policy="naive_overwrite"),

        sharing=SharingConfig(mode="off"),
        poisoning=PoisoningConfig(mode="clean"),

        seeds=[42],
    )





def test_run_logger_initialization(tmp_path: Path, experiment_config: ExperimentConfig) -> None:
    """RunLogger creates run folder and saves config_snapshot.yaml & metadata.json."""
    logger_inst = RunLogger(
        config=experiment_config,
        seed=42,
        base_dir=tmp_path,
        run_id="run_001",
    )

    assert logger_inst.run_dir.exists()
    assert logger_inst.metadata_file.exists()
    assert logger_inst.config_file.exists()

    meta = json.loads(logger_inst.metadata_file.read_text(encoding="utf-8"))
    assert meta["run_id"] == "run_001"
    assert meta["seed"] == 42
    assert meta["experiment_id"] == "test_logging_exp"


def test_run_logger_log_step_and_summary(tmp_path: Path, experiment_config: ExperimentConfig) -> None:
    """log_step appends valid JSONL lines and log_episode_end writes summary.json."""
    logger_inst = RunLogger(
        config=experiment_config,
        seed=42,
        base_dir=tmp_path,
        run_id="run_002",
    )

    logger_inst.log_step(
        round_num=1,
        agent_id="agent_0",
        observation={"pos": [0, 0]},
        prompt="Sample prompt",
        raw_response="Response harvest",
        action="harvest",
        reward=1.0,
        memory_state="Rule A",
        latency_ms=45,
    )

    assert logger_inst.events_file.exists()
    lines = logger_inst.events_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    event = json.loads(lines[0])
    assert event["round"] == 1
    assert event["agent_id"] == "agent_0"
    assert event["action"] == "harvest"
    assert event["reward"] == 1.0
    assert event["latency_ms"] == 45

    logger_inst.log_episode_end(final_score=0.85, summary_info={"total_rewards": 12.0})
    summary_file = logger_inst.run_dir / "summary.json"
    assert summary_file.exists()

    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    assert summary["final_score"] == 0.85
