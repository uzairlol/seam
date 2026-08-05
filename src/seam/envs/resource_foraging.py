"""ResourceForagingGame — 10×10 grid foraging environment."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from seam.envs.base_env import BaseEnv

logger = logging.getLogger(__name__)

_ACTIONS: list[str] = ["stay", "north", "south", "east", "west", "harvest"]

# Row/col deltas for movement actions (row increases southward)
_MOVE_DELTA: dict[str, tuple[int, int]] = {
    "stay": (0, 0),
    "north": (-1, 0),
    "south": (1, 0),
    "east": (0, 1),
    "west": (0, -1),
}


class ResourceForagingGame(BaseEnv):
    """Deterministic multi-agent resource foraging on a 10×10 grid.

    Grid cells hold 0–``max_resources_per_cell`` resource units.  Agents move
    or harvest simultaneously each round.  Collision (two agents on the same
    cell) nullifies any reward that round.

    Args:
        grid_size: Side length of the square grid (default 10).
        n_agents: Number of agents (default 6).
        episode_length: Rounds per episode (default 50).
        resource_spawn_rate: Per-empty-cell spawn probability (default 0.3).
        n_resources_initial: Resources placed at episode start (default 20).
        max_resources_per_cell: Hard cap per cell (default 3).
    """

    def __init__(
        self,
        grid_size: int = 10,
        n_agents: int = 6,
        episode_length: int = 50,
        resource_spawn_rate: float = 0.3,
        n_resources_initial: int = 20,
        max_resources_per_cell: int = 3,
    ) -> None:
        self._grid_size = grid_size
        self._n_agents = n_agents
        self._episode_length = episode_length
        self._resource_spawn_rate = resource_spawn_rate
        self._n_resources_initial = n_resources_initial
        self._max_resources_per_cell = max_resources_per_cell

        # State (initialised in reset())
        self._rng: np.random.Generator = np.random.default_rng()
        self._grid: np.ndarray = np.zeros((grid_size, grid_size), dtype=np.int32)
        self._positions: dict[str, list[int]] = {}  # agent_id -> [row, col]
        self._scores: dict[str, int] = {}
        self._round: int = 0
        self._done: bool = False

        # Tracking for ground-truth score
        self._total_spawned: int = 0
        self._total_harvested: int = 0

    # ------------------------------------------------------------------
    # BaseEnv interface
    # ------------------------------------------------------------------

    @property
    def action_space(self) -> list[str]:
        return list(_ACTIONS)

    @property
    def n_agents(self) -> int:
        return self._n_agents

    def reset(self, seed: int) -> dict[str, Any]:
        """Reset the environment with the given seed.

        Args:
            seed: Controls all random operations for full determinism.

        Returns:
            ``{agent_id: obs_dict}`` with initial observations.
        """
        self._rng = np.random.default_rng(seed)
        self._grid = np.zeros((self._grid_size, self._grid_size), dtype=np.int32)
        self._round = 0
        self._done = False
        self._total_spawned = 0
        self._total_harvested = 0

        # Place initial resources
        flat_indices = self._rng.choice(
            self._grid_size * self._grid_size,
            size=min(self._n_resources_initial, self._grid_size * self._grid_size),
            replace=False,
        )
        for idx in flat_indices:
            r, c = divmod(int(idx), self._grid_size)
            self._grid[r, c] = min(
                self._grid[r, c] + 1, self._max_resources_per_cell
            )
        self._total_spawned += int(self._grid.sum())

        # Place agents at random distinct starting positions
        agent_ids = [f"agent_{i}" for i in range(self._n_agents)]
        positions_flat = self._rng.choice(
            self._grid_size * self._grid_size,
            size=self._n_agents,
            replace=False,
        )
        self._positions = {}
        self._scores = {}
        for agent_id, flat_pos in zip(agent_ids, positions_flat):
            r, c = divmod(int(flat_pos), self._grid_size)
            self._positions[agent_id] = [r, c]
            self._scores[agent_id] = 0

        logger.debug("ResourceForagingGame reset (seed=%d)", seed)
        return {aid: self._build_obs(aid) for aid in agent_ids}

    def step(self, actions: dict[str, Any]) -> dict[str, Any]:
        """Advance the game by one round.

        Action resolution order:
        1. Move all agents simultaneously (clipped at boundaries).
        2. Detect collisions (agents sharing the same cell).
        3. Resolve harvest actions (no reward on collision or empty cell).
        4. Spawn new resources on empty cells.

        Args:
            actions: ``{agent_id: action_str}`` mapping for every agent.

        Returns:
            Standard step result dict with observations, rewards, done, info.

        Raises:
            RuntimeError: If called after the episode is done.
        """
        if self._done:
            raise RuntimeError("step() called on a finished episode — call reset() first.")

        self._round += 1
        agent_ids = list(self._positions.keys())
        rewards: dict[str, float] = {aid: 0.0 for aid in agent_ids}

        # 1. Apply movement
        for aid in agent_ids:
            action = str(actions.get(aid, "stay"))
            if action in _MOVE_DELTA:
                dr, dc = _MOVE_DELTA[action]
                r, c = self._positions[aid]
                new_r = int(np.clip(r + dr, 0, self._grid_size - 1))
                new_c = int(np.clip(c + dc, 0, self._grid_size - 1))
                self._positions[aid] = [new_r, new_c]

        # 2. Detect collisions: cells occupied by more than one agent
        cell_occupants: dict[tuple[int, int], list[str]] = {}
        for aid, (r, c) in [(a, self._positions[a]) for a in agent_ids]:
            cell = (r, c)
            cell_occupants.setdefault(cell, []).append(aid)
        collision_agents: set[str] = set()
        for cell, occupants in cell_occupants.items():
            if len(occupants) > 1:
                collision_agents.update(occupants)

        # 3. Resolve harvests
        # Group harvest requests by cell; only one harvester per cell succeeds
        harvest_requests: dict[tuple[int, int], list[str]] = {}
        for aid in agent_ids:
            if str(actions.get(aid, "stay")) == "harvest":
                cell = (self._positions[aid][0], self._positions[aid][1])
                harvest_requests.setdefault(cell, []).append(aid)

        for cell, harvesters in harvest_requests.items():
            r, c = cell
            # Collision on this cell → no reward for any harvester there
            if cell in {(self._positions[a][0], self._positions[a][1]) for a in collision_agents}:
                continue
            # Must have resources and exactly one harvester (first come, first served)
            if self._grid[r, c] > 0 and len(harvesters) == 1:
                harvester = harvesters[0]
                self._grid[r, c] -= 1
                rewards[harvester] = 1.0
                self._scores[harvester] += 1
                self._total_harvested += 1
            elif self._grid[r, c] > 0 and len(harvesters) > 1:
                # Multiple agents tried to harvest same cell → none succeed
                pass

        # 4. Spawn new resources on empty cells
        spawned_this_round = 0
        for r in range(self._grid_size):
            for c in range(self._grid_size):
                if self._grid[r, c] == 0:
                    if self._rng.random() < self._resource_spawn_rate:
                        self._grid[r, c] = 1
                        spawned_this_round += 1
        self._total_spawned += spawned_this_round

        # Episode termination
        if self._round >= self._episode_length:
            self._done = True

        observations = {aid: self._build_obs(aid) for aid in agent_ids}
        info: dict[str, Any] = {
            "round": self._round,
            "total_harvested": self._total_harvested,
            "total_spawned": self._total_spawned,
            "collision_agents": list(collision_agents),
            "spawned_this_round": spawned_this_round,
        }
        logger.debug(
            "Round %d/%d — harvested=%d spawned_total=%d",
            self._round,
            self._episode_length,
            self._total_harvested,
            self._total_spawned,
        )
        return {
            "observations": observations,
            "rewards": rewards,
            "done": self._done,
            "info": info,
        }

    def get_ground_truth_score(self) -> float:
        """Return the efficiency ratio: harvested / spawned.

        Returns:
            A float in [0.0, 1.0].  Returns 0.0 if nothing was spawned.
        """
        if self._total_spawned == 0:
            return 0.0
        return float(self._total_harvested) / float(self._total_spawned)

    def render(self) -> str:
        """Return an ASCII representation of the current grid state.

        Returns:
            Multi-line string with grid, agent positions, and scores.
        """
        lines: list[str] = [f"=== Round {self._round}/{self._episode_length} ==="]
        # Build overlay: cell content + agent markers
        display = [["." if self._grid[r, c] == 0 else str(self._grid[r, c])
                    for c in range(self._grid_size)]
                   for r in range(self._grid_size)]
        for aid, (r, c) in self._positions.items():
            display[r][c] = "A"  # agent marker overrides resource display
        for row in display:
            lines.append(" ".join(row))
        lines.append("Scores: " + ", ".join(
            f"{aid}={score}" for aid, score in self._scores.items()
        ))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_obs(self, agent_id: str) -> dict[str, Any]:
        """Construct the observation dict for a single agent.

        The observation includes the agent's position and all resources
        visible within a 5×5 neighbourhood (2-cell radius).

        Args:
            agent_id: The agent whose observation to construct.

        Returns:
            Observation dict matching the SEAM spec.
        """
        r, c = self._positions[agent_id]
        visible: list[dict[str, int]] = []
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self._grid_size and 0 <= nc < self._grid_size:
                    qty = int(self._grid[nr, nc])
                    if qty > 0:
                        visible.append({"row": nr, "col": nc, "quantity": qty})
        return {
            "round": self._round,
            "my_position": [r, c],
            "visible_resources": visible,
            "my_score": self._scores[agent_id],
            "rounds_remaining": self._episode_length - self._round,
            "n_agents": self._n_agents,
        }
