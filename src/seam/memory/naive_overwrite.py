"""Naive Overwrite memory policy implementation."""

from __future__ import annotations

import logging
from typing import Any

from seam.agents.decoding import OllamaClient
from seam.memory.base_memory import BaseMemoryPolicy

logger = logging.getLogger(__name__)


class NaiveOverwritePolicy(BaseMemoryPolicy):
    """Naive Overwrite memory policy.

    After each round, the entire memory state is completely rewritten by the LLM
    or replaced with a fresh summary of the recent experience.

    Args:
        max_tokens: Maximum token length limit for memory storage (default 256).
        initial_memory: Optional initial memory string.
    """

    def __init__(self, max_tokens: int = 256, initial_memory: str = "") -> None:
        self.max_tokens = max_tokens
        self._memory_text: str = initial_memory.strip()

    def reset(self) -> None:
        """Clear memory text."""
        self._memory_text = ""

    def get_context(self) -> str:
        """Return current memory string."""
        return self._memory_text

    def update(
        self,
        step_experience: dict[str, Any],
        shared_context: str = "",
        client: OllamaClient | None = None,
    ) -> str:
        """Overwrite current memory with a reflection on the latest experience and shared peer context.

        Args:
            step_experience: Dict with keys ``"observation"``, ``"action"``, ``"reward"``.
            shared_context: Incoming peer memory text snippets.
            client: Optional LLM client for generating reflection.

        Returns:
            Updated memory text.
        """
        obs = step_experience.get("observation", {})
        action = step_experience.get("action", "")
        reward = step_experience.get("reward", 0.0)

        shared_str = f"\n\n{shared_context}" if shared_context.strip() else ""
        prompt = (
            "=== Previous Memory ===\n"
            f"{self._memory_text if self._memory_text else 'None'}\n\n"
            "=== Recent Experience ===\n"
            f"Observation: {obs}\n"
            f"Action Taken: {action}\n"
            f"Reward Received: {reward}"
            f"{shared_str}\n\n"
            "=== Task ===\n"
            "Completely rewrite your memory reflection to guide future decisions based on your experience and shared peer memory. "
            "Keep it concise and actionable under 100 words."
        )

        if client is not None:
            try:
                new_mem, _ = client.complete(prompt)
                self._memory_text = new_mem.strip()
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM call failed during naive overwrite reflection: %s", exc)
                self._memory_text = f"Last Action: {action} | Reward: {reward}"
        else:
            # Deterministic fallback when no LLM client is provided
            base_mem = f"Last Action: {action} | Reward: {reward}"
            if shared_context.strip():
                peer_lines = [l.strip() for l in shared_context.splitlines() if l.strip() and not l.strip().startswith("===")]
                if peer_lines:
                    base_mem += " | Peer Memory: " + "; ".join(peer_lines)
            self._memory_text = base_mem

        return self._memory_text

    def to_dict(self) -> dict[str, Any]:
        """Serialize state."""
        return {
            "policy": "naive_overwrite",
            "max_tokens": self.max_tokens,
            "memory_text": self._memory_text,
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        """Restore state."""
        self.max_tokens = data.get("max_tokens", 256)
        self._memory_text = str(data.get("memory_text", ""))
