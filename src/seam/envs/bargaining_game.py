"""BargainingGame — random-pair ultimatum bargaining environment."""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

from seam.envs.base_env import BaseEnv

logger = logging.getLogger(__name__)

# Valid responder actions
_RESPONDER_ACTIONS: list[str] = ["accept", "reject"]
_SPLIT_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s*$")


class BargainingGame(BaseEnv):
    """Ultimatum bargaining between randomly paired agents.

    Each round, one randomly selected proposer submits a split of
    ``pie_size`` units (format: ``"<own> <other>"``).  A randomly chosen
    responder accepts or rejects.  Accept → both receive their share.
    Reject → both receive 0.  Pairs are selected uniformly at random.

    Nash equilibrium is a 50–50 split.

    Args:
        n_agents: Number of agents (default 6).
        episode_length: Rounds per episode (default 40).
        pie_size: Total units to split each round (default 100).
    """

    def __init__(
        self,
        n_agents: int = 6,
        episode_length: int = 40,
        pie_size: int = 100,
    ) -> None:
        self._n_agents = n_agents
        self._episode_length = episode_length
        self._pie_size = pie_size

        self._rng: np.random.Generator = np.random.default_rng()
        self._round: int = 0
        self._done: bool = False
        self._scores: dict[str, float] = {}
        # History of accepted deals: list of (proposer_share, responder_share)
        self._accepted_deals: list[tuple[int, int]] = []

    # ------------------------------------------------------------------
    # BaseEnv interface
    # ------------------------------------------------------------------

    @property
    def action_space(self) -> list[str]:
        """Proposer actions: ``"<own> <other>"`` strings; responder: accept/reject."""
        # For logging purposes, list the canonical responder actions plus a
        # representative proposer format hint.
        return _RESPONDER_ACTIONS + [f"{i} {self._pie_size - i}" for i in range(self._pie_size + 1)]

    @property
    def n_agents(self) -> int:
        return self._n_agents

    def reset(self, seed: int) -> dict[str, Any]:
        """Reset the environment.

        Args:
            seed: Seed for all random operations.

        Returns:
            ``{agent_id: obs_dict}`` with initial observations.
        """
        self._rng = np.random.default_rng(seed)
        self._round = 0
        self._done = False
        self._accepted_deals = []
        agent_ids = [f"agent_{i}" for i in range(self._n_agents)]
        self._scores = {aid: 0.0 for aid in agent_ids}
        logger.debug("BargainingGame reset (seed=%d, pie=%d)", seed, self._pie_size)
        return {aid: self._build_obs(aid, role="observer") for aid in agent_ids}

    def step(self, actions: dict[str, Any]) -> dict[str, Any]:
        """Advance one bargaining round.

        Selects a random proposer–responder pair from all agents.  Their
        submitted actions are used to resolve the deal.

        Args:
            actions: ``{agent_id: action_str}``.  The proposer's action should
                be ``"<own> <other>"``; the responder's should be
                ``"accept"`` or ``"reject"``.

        Returns:
            Standard step dict with observations, rewards, done, info.

        Raises:
            RuntimeError: If called after the episode is done.
        """
        if self._done:
            raise RuntimeError("step() called on a finished episode — call reset() first.")

        self._round += 1
        agent_ids = list(self._scores.keys())
        rewards: dict[str, float] = {aid: 0.0 for aid in agent_ids}

        # Randomly select proposer and responder (distinct)
        indices = self._rng.choice(self._n_agents, size=2, replace=False)
        proposer_id = agent_ids[int(indices[0])]
        responder_id = agent_ids[int(indices[1])]

        # Parse proposer's split
        prop_action = str(actions.get(proposer_id, f"{self._pie_size // 2} {self._pie_size // 2}"))
        own_share, other_share = self._parse_split(prop_action)

        # Parse responder decision
        resp_action = str(actions.get(responder_id, "reject")).strip().lower()
        accepted = resp_action == "accept"

        deal_info: dict[str, Any] = {
            "proposer": proposer_id,
            "responder": responder_id,
            "proposed_split": [own_share, other_share],
            "accepted": accepted,
        }

        if accepted:
            rewards[proposer_id] = float(own_share)
            rewards[responder_id] = float(other_share)
            self._scores[proposer_id] += own_share
            self._scores[responder_id] += other_share
            self._accepted_deals.append((own_share, other_share))
            logger.debug(
                "Round %d: %s proposed %d/%d — accepted by %s",
                self._round, proposer_id, own_share, other_share, responder_id,
            )
        else:
            logger.debug(
                "Round %d: %s proposed %d/%d — rejected by %s",
                self._round, proposer_id, own_share, other_share, responder_id,
            )

        if self._round >= self._episode_length:
            self._done = True

        observations = {
            aid: self._build_obs(
                aid,
                role="proposer" if aid == proposer_id else
                     "responder" if aid == responder_id else "observer",
            )
            for aid in agent_ids
        }
        info: dict[str, Any] = {
            "round": self._round,
            "deal": deal_info,
            "n_accepted_deals": len(self._accepted_deals),
        }
        return {
            "observations": observations,
            "rewards": rewards,
            "done": self._done,
            "info": info,
        }

    def get_ground_truth_score(self) -> float:
        """Return a fairness score: ``1 / (1 + mean_deviation_from_50)``.

        Where ``mean_deviation_from_50`` is the mean absolute deviation of the
        proposer's share from 50 across all accepted deals.

        Returns:
            Float in (0.0, 1.0].  Returns 1.0 (perfect) if no deals accepted.
        """
        if not self._accepted_deals:
            # No accepted deals; return 1.0 (no unfairness observed)
            return 1.0
        deviations = [abs(own - self._pie_size / 2) for own, _ in self._accepted_deals]
        mean_dev = sum(deviations) / len(deviations)
        return 1.0 / (1.0 + mean_dev)

    def render(self) -> str:
        """Return a human-readable game state string.

        Returns:
            Multi-line string with current scores and deal history summary.
        """
        lines = [
            f"=== Round {self._round}/{self._episode_length} ===",
            f"Accepted deals: {len(self._accepted_deals)}",
            f"Score: {self.get_ground_truth_score():.4f}",
            "Cumulative scores:",
        ]
        for aid, score in self._scores.items():
            lines.append(f"  {aid}: {score:.1f}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_split(self, action: str) -> tuple[int, int]:
        """Parse a proposer split action string.

        Valid format: ``"<own> <other>"`` where both values are non-negative
        integers.  Invalid / out-of-budget proposals default to 50–50.

        Args:
            action: Raw proposer action string.

        Returns:
            ``(own_share, other_share)`` tuple summing to ``pie_size``.
        """
        match = _SPLIT_RE.match(action)
        if match:
            own = int(match.group(1))
            other = int(match.group(2))
            if own + other == self._pie_size and own >= 0 and other >= 0:
                return own, other
        logger.warning("Invalid proposer split '%s' — defaulting to 50/50", action)
        half = self._pie_size // 2
        return half, self._pie_size - half

    def _build_obs(self, agent_id: str, role: str = "observer") -> dict[str, Any]:
        """Build an observation dict for one agent.

        Args:
            agent_id: Target agent.
            role: ``"proposer"``, ``"responder"``, or ``"observer"``.

        Returns:
            Observation dict.
        """
        return {
            "round": self._round,
            "rounds_remaining": self._episode_length - self._round,
            "role": role,
            "my_score": self._scores.get(agent_id, 0.0),
            "n_accepted_deals": len(self._accepted_deals),
            "n_agents": self._n_agents,
            "pie_size": self._pie_size,
        }
