"""Unit tests for RunRehydrator parsing run directories into DataFrames."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from seam.logging.rehydrator import RunRehydrator
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
def sample_run_dir(tmp_path: Path) -> Path:
    cfg = ExperimentConfig(
        experiment_id="test_rehydrate_exp",
        description="Rehydrator unit test",
        env=EnvConfig(type="resource_foraging"),
        model=ModelConfig(model_name="qwen2.5:7b-instruct", base_url="http://localhost:11434"),
        memory=MemoryConfig(policy="naive_overwrite"),
        sharing=SharingConfig(mode="off"),
        poisoning=PoisoningConfig(mode="clean"),
        seeds=[42],
    )

    logger_inst = RunLogger(config=cfg, seed=42, base_dir=tmp_path, run_id="run_rehydrate_test")
    logger_inst.log_step(
        round_num=1,
        agent_id="agent_0",
        observation={"pos": [0, 0]},
        prompt="p1",
        raw_response="r1",
        action="harvest",
        reward=1.0,
        memory_state="mem1",
    )
    logger_inst.log_step(
        round_num=1,
        agent_id="agent_1",
        observation={"pos": [1, 1]},
        prompt="p2",
        raw_response="r2",
        action="stay",
        reward=0.0,
        memory_state="mem2",
    )
    logger_inst.log_episode_end(final_score=0.50)
    return logger_inst.run_dir


def test_rehydrator_load_metadata_and_config(sample_run_dir: Path) -> None:
    """Rehydrator correctly loads metadata and config snapshot dicts."""
    rehydrator = RunRehydrator(sample_run_dir)
    meta = rehydrator.load_metadata()
    cfg_dict = rehydrator.load_config()

    assert meta["run_id"] == "run_rehydrate_test"
    assert meta["seed"] == 42
    assert cfg_dict["experiment_id"] == "test_rehydrate_exp"


def test_rehydrator_to_dataframe(sample_run_dir: Path) -> None:
    """Rehydrator converts step events to pandas DataFrame."""
    rehydrator = RunRehydrator(sample_run_dir)
    df = rehydrator.to_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df["agent_id"]) == ["agent_0", "agent_1"]
    assert list(df["action"]) == ["harvest", "stay"]
    assert list(df["reward"]) == [1.0, 0.0]
