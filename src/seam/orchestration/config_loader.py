"""Pydantic-based config loader for SEAM experiment configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from seam.utils.io import load_yaml


class ModelConfig(BaseModel):
    """Configuration for the Ollama model used by agents."""

    model_name: str
    base_url: str
    temperature: float = 0.0
    seed: Optional[int] = None
    max_tokens: int = 512
    top_p: float = 1.0
    request_timeout: int = 60
    retry_attempts: int = 3


class EnvConfig(BaseModel):
    """Configuration for the task environment."""

    type: str
    grid_size: int = 10
    n_agents: int = 6
    episode_length: int = 50
    resource_spawn_rate: float = 0.3


class MemoryConfig(BaseModel):
    """Configuration for an agent memory policy."""

    policy: str  # "naive_overwrite" | "raw_buffer" | "structured_update"
    max_tokens: int = 256
    window_size: int = 10  # for raw_buffer
    max_playbook_entries: int = 20  # for structured_update


class SharingConfig(BaseModel):
    """Configuration for the shared broadcast memory channel."""

    mode: str  # "off" | "broadcast" | "selective" | "peer_to_peer"
    topology: str = "full_broadcast"  # "full_broadcast" | "ring" | "star" | "cluster" | "off"
    publish_every_n_rounds: int = 3
    max_artifact_tokens: int = 128
    consume_mode: str = "all"  # "all" | "top_k"


class PoisoningConfig(BaseModel):
    """Configuration for memory poisoning injection."""

    mode: str  # "clean" | "poisoned"
    poison_agent_id: str = "agent_0"
    injection_mode: str = "internal"  # "internal" | "channel" | "gradual"
    poison_file: str = ""


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration combining all sub-configs."""

    experiment_id: str
    description: str
    env: EnvConfig
    model: ModelConfig
    memory: MemoryConfig
    sharing: SharingConfig
    poisoning: PoisoningConfig
    seeds: list[int]


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment YAML file and parse it into an ExperimentConfig.

    Args:
        path: Path to the experiment YAML config file.

    Returns:
        A validated ExperimentConfig instance.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        pydantic.ValidationError: If the YAML does not match the schema.
    """
    raw = load_yaml(path)
    return ExperimentConfig.model_validate(raw)
