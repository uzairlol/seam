"""Raw Trajectory Buffer memory policy implementation."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from seam.agents.decoding import OllamaClient
from seam.memory.base_memory import BaseMemoryPolicy

logger = logging.getLogger(__name__)


class RawTrajectoryBufferPolicy(BaseMemoryPolicy):
    """Raw Trajectory Buffer memory policy.

    Maintains a sliding window of raw ``(state, action, reward)`` experience tuples
    without LLM reflection or compression.

    Args:
        window_size: Maximum number of recent step experiences to retain (default 10).
    """

    def __init__(self, window_size: int = 10) -> None:
        self.window_size = window_size
        self._buffer: deque[dict[str, Any]] = deque(maxlen=window_size)

    def reset(self) -> None:
        """Clear trajectory buffer."""
        self._buffer.clear()

    def get_context(self) -> str:
        """Format stored trajectory buffer as chronological text lines.

        Returns:
            Formatted history string.
        """
        if not self._buffer:
            return ""

        lines = ["=== Recent Experience Trajectory ==="]
        for idx, exp in enumerate(self._buffer, start=1):
            obs = exp.get("observation", {})
            act = exp.get("action", "")
            rew = exp.get("reward", 0.0)
            lines.append(f"Step {idx}: Obs={obs} -> Action='{act}' -> Reward={rew:.2f}")

        return "\n".join(lines)

    def update(
        self,
        step_experience: dict[str, Any],
        shared_context: str = "",
        client: OllamaClient | None = None,
    ) -> str:
        """Append latest step experience and shared peer context to sliding window buffer.

        Args:
            step_experience: Dict with keys ``"observation"``, ``"action"``, ``"reward"``.
            shared_context: Incoming peer memory text snippets.
            client: Unused for raw trajectory policy (no LLM reflection).

        Returns:
            Formatted memory context.
        """
        # Store clean copy of experience
        entry = {
            "observation": step_experience.get("observation", {}),
            "action": str(step_experience.get("action", "")),
            "reward": float(step_experience.get("reward", 0.0)),
        }
        if shared_context.strip():
            entry["shared_peer_context"] = shared_context.strip()
        self._buffer.append(entry)
        return self.get_context()

    def to_dict(self) -> dict[str, Any]:
        """Serialize state."""
        return {
            "policy": "raw_trajectory_buffer",
            "window_size": self.window_size,
            "buffer": list(self._buffer),
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        """Restore state."""
        self.window_size = data.get("window_size", 10)
        buffer_list = data.get("buffer", [])
        self._buffer = deque(buffer_list, maxlen=self.window_size)
