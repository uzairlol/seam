"""Abstract base class for all SEAM task environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEnv(ABC):
    """Abstract interface for all SEAM task environments.

    Every environment must be fully deterministic given the same seed.
    Rewards are objective and require no LLM judge.
    """

    @abstractmethod
    def reset(self, seed: int) -> dict[str, Any]:
        """Reset the environment to an initial state.

        Args:
            seed: Integer seed for all random operations in this episode.

        Returns:
            A mapping of ``{agent_id: initial_obs_dict}`` for all agents.
        """

    @abstractmethod
    def step(self, actions: dict[str, Any]) -> dict[str, Any]:
        """Advance the environment by one round.

        Args:
            actions: Mapping of ``{agent_id: action}`` for all agents.

        Returns:
            A dict with the following keys:

            - ``"observations"``: ``{agent_id: obs_dict}``
            - ``"rewards"``: ``{agent_id: float}``
            - ``"done"``: ``bool`` — True when the episode has ended
            - ``"info"``: ``dict`` — auxiliary diagnostics
        """

    @abstractmethod
    def get_ground_truth_score(self) -> float:
        """Compute an objective, unambiguous episode score.

        The score must not require an LLM judge.  Higher is always better.

        Returns:
            A scalar score (typically in [0.0, 1.0]).
        """

    @abstractmethod
    def render(self) -> str:
        """Produce a human-readable snapshot of the current state.

        Returns:
            A multi-line string suitable for logging or debugging.
        """

    @property
    @abstractmethod
    def action_space(self) -> list[str]:
        """List of valid action strings available to agents."""

    @property
    @abstractmethod
    def n_agents(self) -> int:
        """Number of agents participating in this environment."""
