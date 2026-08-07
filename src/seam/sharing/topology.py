"""TopologyGenerator for defining inter-agent communication channels and networks."""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_TOPOLOGIES = {"full_broadcast", "broadcast", "ring", "star", "cluster", "off"}


class TopologyGenerator:
    """Generates adjacency topologies determining which agents can exchange memory updates.

    Args:
        n_agents: Number of agents in the population.
        topology_type: One of ``"full_broadcast"`` (or ``"broadcast"``), ``"ring"``,
            ``"star"``, ``"cluster"``, or ``"off"``.
        seed: Random seed for stochastic topologies (e.g. cluster/random).
    """

    def __init__(self, n_agents: int, topology_type: str = "full_broadcast", seed: int | None = None) -> None:
        self.n_agents = n_agents
        self.topology_type = topology_type.lower().strip()
        self.seed = seed
        self.agent_ids = [f"agent_{i}" for i in range(n_agents)]

        if self.topology_type not in SUPPORTED_TOPOLOGIES:
            raise ValueError(
                f"Unsupported topology '{topology_type}'. Supported: {sorted(SUPPORTED_TOPOLOGIES)}"
            )

        self._adj_matrix = self._build_adjacency_matrix()

    def _build_adjacency_matrix(self) -> np.ndarray:
        """Construct a binary NxN adjacency matrix where entry (i, j) == 1 if agent i receives from agent j."""
        n = self.n_agents
        matrix = np.zeros((n, n), dtype=int)

        if self.topology_type in ("full_broadcast", "broadcast"):
            # Everyone receives from everyone except self (off-diagonal 1s)
            matrix = np.ones((n, n), dtype=int)
            np.fill_diagonal(matrix, 0)

        elif self.topology_type == "ring":
            # Directed ring: agent_i receives from agent_{(i-1)%n} and agent_{(i+1)%n}
            for i in range(n):
                matrix[i, (i - 1) % n] = 1
                matrix[i, (i + 1) % n] = 1

        elif self.topology_type == "star":
            # Hub is agent_0. Hub communicates with all leaves; leaves communicate only with hub.
            for i in range(1, n):
                matrix[0, i] = 1
                matrix[i, 0] = 1

        elif self.topology_type == "cluster":
            # 2 clusters (split agents in half). Intra-cluster full broadcast.
            mid = max(1, n // 2)
            # Cluster 1: [0, mid)
            for i in range(mid):
                for j in range(mid):
                    if i != j:
                        matrix[i, j] = 1
            # Cluster 2: [mid, n)
            for i in range(mid, n):
                for j in range(mid, n):
                    if i != j:
                        matrix[i, j] = 1

        elif self.topology_type == "off":
            # No connections
            matrix = np.zeros((n, n), dtype=int)

        return matrix

    @property
    def adjacency_matrix(self) -> np.ndarray:
        """Return the NxN adjacency matrix."""
        return self._adj_matrix

    def get_neighbors(self, agent_id: str) -> list[str]:
        """Return list of agent IDs from which *agent_id* receives memory updates.

        Args:
            agent_id: Target agent identifier string (e.g. ``"agent_1"``).

        Returns:
            List of neighbor agent ID strings.
        """
        if agent_id not in self.agent_ids:
            raise KeyError(f"Agent '{agent_id}' not found in topology (known: {self.agent_ids})")

        idx = self.agent_ids.index(agent_id)
        neighbor_indices = np.where(self._adj_matrix[idx] == 1)[0]
        return [self.agent_ids[i] for i in neighbor_indices]
