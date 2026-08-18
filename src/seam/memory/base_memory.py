"""Abstract base class for all SEAM memory policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from seam.agents.decoding import OllamaClient


class BaseMemoryPolicy(ABC):
    """Abstract base class for agent memory evolution policies.

    Every memory policy manages state persistent across rounds within an episode.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset internal memory state to empty."""

    @abstractmethod
    def update(
        self,
        step_experience: dict[str, Any],
        shared_context: str = "",
        client: OllamaClient | None = None,
    ) -> str:
        """Update memory given a new round experience step.

        Args:
            step_experience: Dict containing ``"observation"``, ``"action"``, ``"reward"``, etc.
            client: Optional :class:`OllamaClient` for memory reflection/synthesis.

        Returns:
            The updated memory context string.
        """

    @abstractmethod
    def get_context(self) -> str:
        """Return the current memory text context for inclusion in agent prompts."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize memory state to a JSON-compatible dictionary."""

    @abstractmethod
    def from_dict(self, data: dict[str, Any]) -> None:
        """Restore memory state from a dictionary snapshot."""
