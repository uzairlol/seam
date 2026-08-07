"""Orchestration module exports."""

from seam.orchestration.config_loader import (
    EnvConfig,
    ExperimentConfig,
    MemoryConfig,
    ModelConfig,
    PoisoningConfig,
    SharingConfig,
    load_experiment_config,
)

__all__ = [
    "EnvConfig",
    "EpisodeRunner",
    "ExperimentConfig",
    "MemoryConfig",
    "ModelConfig",
    "PoisoningConfig",
    "SharingConfig",
    "load_experiment_config",
]


def __getattr__(name: str):  # type: ignore[override]
    """Lazy-load heavy submodules to break circular imports at collection time."""
    if name == "EpisodeRunner":
        from seam.orchestration.runner import EpisodeRunner  # noqa: PLC0415
        return EpisodeRunner
    raise AttributeError(f"module 'seam.orchestration' has no attribute {name!r}")
