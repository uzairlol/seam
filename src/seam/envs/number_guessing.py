"""NumberGuessingGame — multi-agent binary-search validation environment."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from seam.envs.base_env import BaseEnv

logger = logging.getLogger(__name__)

_VALID_RANGE = range(1, 101)
_ACTIONS: list[str] = [str(i) for i in _VALID_RANGE]


class NumberGuessingGame(BaseEnv):
    """Cooperative number guessing environment.

    A secret number in ``[secret_min, secret_max]`` is drawn once at episode
    start.  Each round every agent independently submits a guess (an integer
    string).  The environment returns ``"higher"``, ``"lower"``, or
    ``"correct"`` per agent based on the relationship to the secret.  When any
    agent guesses correctly, *all* agents receive +1 reward and the episode
    ends.

    Args:
        n_agents: Number of agents (default 6).
        episode_length: Maximum rounds before the episode ends (default 30).
        secret_min: Lower bound of the secret (inclusive, default 1).
        secret_max: Upper bound of the secret (inclusive, default 100).
    """

    def __init__(
        self,
        n_agents: int = 6,
        episode_length: int = 30,
        secret_min: int = 1,
        secret_max: int = 100,
    ) -> None:
        self._n_agents = n_agents
        self._episode_length = episode_length
        self._secret_min = secret_min
        self._secret_max = secret_max

        self._rng: np.random.Generator = np.random.default_rng()
        self._secret: int = secret_min
        self._round: int = 0
        self._done: bool = False
        self._rounds_to_solve: int | None = None
        # Per-agent feedback history: list of (guess, feedback) tuples
        self._histories: dict[str, list[tuple[int, str]]] = {}

    # ------------------------------------------------------------------
    # BaseEnv interface
    # ------------------------------------------------------------------

    @property
    def action_space(self) -> list[str]:
        """Integer strings from '1' to '100' (inclusive)."""
        return [str(i) for i in range(self._secret_min, self._secret_max + 1)]

    @property
    def n_agents(self) -> int:
        return self._n_agents

    def reset(self, seed: int) -> dict[str, Any]:
        """Reset the environment.

        Args:
            seed: Seed for deterministic secret selection.

        Returns:
            ``{agent_id: obs_dict}`` with initial observations.
        """
        self._rng = np.random.default_rng(seed)
        self._secret = int(self._rng.integers(self._secret_min, self._secret_max + 1))
        self._round = 0
        self._done = False
        self._rounds_to_solve = None
        agent_ids = [f"agent_{i}" for i in range(self._n_agents)]
        self._histories = {aid: [] for aid in agent_ids}
        logger.debug("NumberGuessingGame reset (seed=%d, secret=%d)", seed, self._secret)
        return {aid: self._build_obs(aid) for aid in agent_ids}

    def step(self, actions: dict[str, Any]) -> dict[str, Any]:
        """Advance one guessing round.

        Args:
            actions: ``{agent_id: guess_str}`` where each guess is an integer
                string in the valid range.  Out-of-range guesses are treated as
                ``"lower"`` if too high, ``"higher"`` if too low.

        Returns:
            Standard step result with observations, rewards, done flag, info.

        Raises:
            RuntimeError: If called after the episode is done.
        """
        if self._done:
            raise RuntimeError("step() called on a finished episode — call reset() first.")

        self._round += 1
        agent_ids = list(self._histories.keys())
        rewards: dict[str, float] = {aid: 0.0 for aid in agent_ids}
        feedbacks: dict[str, str] = {}

        # Evaluate guesses
        any_correct = False
        for aid in agent_ids:
            raw = str(actions.get(aid, str(self._secret_min)))
            try:
                guess = int(raw)
            except ValueError:
                guess = self._secret_min  # fallback for invalid strings

            if guess < self._secret:
                feedback = "higher"
            elif guess > self._secret:
                feedback = "lower"
            else:
                feedback = "correct"
                any_correct = True

            feedbacks[aid] = feedback
            self._histories[aid].append((guess, feedback))

        # Reward all agents if any got it right
        if any_correct:
            for aid in agent_ids:
                rewards[aid] = 1.0
            self._rounds_to_solve = self._round
            self._done = True

        if self._round >= self._episode_length:
            self._done = True

        observations = {aid: self._build_obs(aid) for aid in agent_ids}
        info: dict[str, Any] = {
            "round": self._round,
            "feedbacks": feedbacks,
            "any_correct": any_correct,
            "secret": self._secret if self._done else None,  # reveal only at end
        }
        return {
            "observations": observations,
            "rewards": rewards,
            "done": self._done,
            "info": info,
        }

    def get_ground_truth_score(self) -> float:
        """Return ``1 / rounds_to_solve``, or 0.0 if unsolved.

        Returns:
            Float in (0.0, 1.0] if solved, 0.0 otherwise.
        """
        if self._rounds_to_solve is None:
            return 0.0
        return 1.0 / self._rounds_to_solve

    def render(self) -> str:
        """Return a human-readable game state string.

        Returns:
            Multi-line string with round info and each agent's guess history.
        """
        lines = [f"=== Round {self._round}/{self._episode_length} ==="]
        for aid, history in self._histories.items():
            if history:
                last_guess, last_feedback = history[-1]
                lines.append(f"  {aid}: last guess={last_guess} → {last_feedback}")
            else:
                lines.append(f"  {aid}: no guesses yet")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_obs(self, agent_id: str) -> dict[str, Any]:
        """Build observation for one agent.

        Args:
            agent_id: Target agent.

        Returns:
            Observation dict with guess history and remaining rounds.
        """
        return {
            "round": self._round,
            "rounds_remaining": self._episode_length - self._round,
            "history": list(self._histories.get(agent_id, [])),
            "n_agents": self._n_agents,
        }
