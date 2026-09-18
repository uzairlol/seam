"""NoMemoryPolicy — zero-shot baseline memory policy that maintains no state across rounds."""

from __future__ import annotations

from typing import Any

from seam.agents.decoding import OllamaClient
from seam.memory.base_memory import BaseMemoryPolicy


class NoMemoryPolicy(BaseMemoryPolicy):
    """Zero-shot baseline memory policy that stores no history.

    Always returns an empty memory context, serving as an experimental control
    to isolate the benefit of persistent memory vs zero-shot performance.
    """

    def reset(self) -> None:
        """No-op: no internal memory state to reset."""

    def update(
        self,
        step_experience: dict[str, Any],
        shared_context: str = "",
        client: OllamaClient | None = None,
    ) -> str:
        """No-op: memory is not updated and returns empty context."""
        return ""

    def get_context(self) -> str:
        """Return empty string indicating no stored memory context."""
        return ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize memory state (empty)."""
        return {"policy": "no_memory"}

    def from_dict(self, data: dict[str, Any]) -> None:
        """Restore memory state (no-op)."""
