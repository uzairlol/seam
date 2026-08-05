"""Tests for config loading via ExperimentConfig and load_experiment_config."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from seam.orchestration.config_loader import (
    ExperimentConfig,
    load_experiment_config,
)


SAMPLE_YAML = textwrap.dedent("""\
    experiment_id: "test_exp_001"
    description: "Unit test experiment"
    env:
      type: "resource_foraging"
      grid_size: 10
      n_agents: 6
      episode_length: 50
      resource_spawn_rate: 0.3
    model:
      model_name: "qwen2.5:7b-instruct"
      base_url: "http://localhost:11434"
      temperature: 0.0
      seed: 42
      max_tokens: 512
      top_p: 1.0
      request_timeout: 60
      retry_attempts: 3
    memory:
      policy: "naive_overwrite"
      max_tokens: 256
      window_size: 10
      max_playbook_entries: 20
    sharing:
      mode: "off"
      publish_every_n_rounds: 3
      max_artifact_tokens: 128
      consume_mode: "all"
    poisoning:
      mode: "clean"
      poison_agent_id: "agent_0"
      injection_mode: "internal"
      poison_file: ""
    seeds: [1, 2, 3]
""")


@pytest.fixture
def sample_yaml_path(tmp_path: Path) -> Path:
    """Write the sample YAML to a temp file and return its path."""
    config_file = tmp_path / "test_experiment.yaml"
    config_file.write_text(SAMPLE_YAML, encoding="utf-8")
    return config_file


def test_load_experiment_config_returns_correct_type(sample_yaml_path: Path) -> None:
    """load_experiment_config should return an ExperimentConfig instance."""
    cfg = load_experiment_config(sample_yaml_path)
    assert isinstance(cfg, ExperimentConfig)


def test_load_experiment_config_values(sample_yaml_path: Path) -> None:
    """All fields should be parsed correctly from the YAML."""
    cfg = load_experiment_config(sample_yaml_path)
    assert cfg.experiment_id == "test_exp_001"
    assert cfg.env.grid_size == 10
    assert cfg.env.n_agents == 6
    assert cfg.model.model_name == "qwen2.5:7b-instruct"
    assert cfg.model.seed == 42
    assert cfg.memory.policy == "naive_overwrite"
    assert cfg.sharing.mode == "off"
    assert cfg.poisoning.mode == "clean"
    assert cfg.seeds == [1, 2, 3]


def test_load_experiment_config_defaults(sample_yaml_path: Path) -> None:
    """Default values should be applied for fields not in YAML."""
    cfg = load_experiment_config(sample_yaml_path)
    assert cfg.model.temperature == 0.0
    assert cfg.model.top_p == 1.0
    assert cfg.memory.window_size == 10


def test_load_experiment_config_missing_file() -> None:
    """load_experiment_config should raise FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        load_experiment_config("/nonexistent/path/to/config.yaml")


def test_experiment_config_validation_error(tmp_path: Path) -> None:
    """ExperimentConfig should raise ValidationError if required fields are missing."""
    import pydantic

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("experiment_id: 'no_other_fields'\n", encoding="utf-8")
    with pytest.raises(pydantic.ValidationError):
        load_experiment_config(bad_yaml)
