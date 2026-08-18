"""Structured RunLogger for recording append-only JSONL experiment logs and metadata."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from seam.orchestration.config_loader import ExperimentConfig
from seam.utils.io import save_json, save_yaml

logger = logging.getLogger(__name__)


class RunLogger:
    """Manages experiment logging directory, config snapshots, and append-only event logs.

    Args:
        config: The :class:`ExperimentConfig` for the run.
        seed: Random seed used for this run.
        base_dir: Parent directory where run folders are created (default ``"runs"``).
        run_id: Optional explicit run identifier string.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        seed: int,
        base_dir: str | Path = "runs",
        run_id: str | None = None,
    ) -> None:
        self.config = config
        self.seed = seed
        self.timestamp = datetime.now(timezone.utc).isoformat()

        if run_id:
            self.run_id = run_id
        else:
            ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self.run_id = f"{config.experiment_id}_seed{seed}_{ts_str}"

        self.run_dir = Path(base_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.events_file = self.run_dir / "events.jsonl"
        self.metadata_file = self.run_dir / "metadata.json"
        self.config_file = self.run_dir / "config_snapshot.yaml"

        self._init_run_directory()

    def _init_run_directory(self) -> None:
        """Save initial metadata and config snapshot."""
        metadata = {
            "run_id": self.run_id,
            "experiment_id": self.config.experiment_id,
            "seed": self.seed,
            "timestamp": self.timestamp,
            "env_type": self.config.env.type,
            "n_agents": self.config.env.n_agents,
            "model_name": self.config.model.model_name,
            "memory_policy": self.config.memory.policy,
            "sharing_mode": self.config.sharing.mode,
            "topology": self.config.sharing.topology,
            "poisoning_mode": self.config.poisoning.mode,
        }
        save_json(metadata, self.metadata_file)
        save_yaml(self.config.model_dump(), self.config_file)
        logger.info("Initialised run directory: %s", self.run_dir)

    def log_step(
        self,
        round_num: int,
        agent_id: str,
        observation: dict[str, Any],
        prompt: str,
        raw_response: str,
        action: str,
        reward: float,
        memory_state: str,
        info: dict[str, Any] | None = None,
        latency_ms: int = 0,
    ) -> dict[str, Any]:
        """Append a single agent round event to the jsonl log file.

        Args:
            round_num: Episode round number (1-indexed).
            agent_id: Identifier of the acting agent.
            observation: Environment observation dict.
            prompt: Assembled prompt text sent to LLM.
            raw_response: Raw text returned by LLM.
            action: Parsed action string.
            reward: Scalar reward received for action.
            memory_state: Active memory context string.
            info: Optional additional step diagnostic info.
            latency_ms: LLM call wall-clock latency in milliseconds.

        Returns:
            The recorded event dictionary.
        """
        event = {
            "round": round_num,
            "agent_id": agent_id,
            "observation": observation,
            "prompt": prompt,
            "raw_response": raw_response,
            "action": action,
            "reward": float(reward),
            "memory_state": memory_state,
            "latency_ms": latency_ms,
            "info": info or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with self.events_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

        return event

    def log_episode_end(self, final_score: float, summary_info: dict[str, Any] | None = None) -> None:
        """Record final summary metadata at episode completion.

        Args:
            final_score: Objective episode score.
            summary_info: Dict of overall episode stats.
        """
        summary_file = self.run_dir / "summary.json"
        summary_data = {
            "run_id": self.run_id,
            "final_score": final_score,
            "summary_info": summary_info or {},
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        save_json(summary_data, summary_file)
        logger.info("Completed run %s — final score: %.4f", self.run_id, final_score)
