"""MemorySharingEngine managing inter-agent memory artifact exchange and broadcast channels."""

from __future__ import annotations

import logging
from typing import Any

from seam.memory.base_memory import BaseMemoryPolicy
from seam.orchestration.config_loader import SharingConfig
from seam.sharing.topology import TopologyGenerator

logger = logging.getLogger(__name__)


class MemorySharingEngine:
    """Orchestrates memory publication, topology-directed routing, and shared context formatting.

    Args:
        config: :class:`SharingConfig` specifying sharing parameters (mode, cadence, etc.).
        n_agents: Number of agents participating in the population.
        topology_type: Network topology (e.g. ``"full_broadcast"``, ``"ring"``, ``"star"``, ``"off"``).
    """

    def __init__(
        self,
        config: SharingConfig,
        n_agents: int,
        topology_type: str = "full_broadcast",
    ) -> None:
        self.config = config
        self.n_agents = n_agents
        # If sharing mode is "off", override topology to "off"
        effective_topology = "off" if config.mode == "off" else topology_type
        self.topology = TopologyGenerator(n_agents=n_agents, topology_type=effective_topology)

        # Inboxes store incoming shared memory context string snippets for each agent
        self.agent_ids = [f"agent_{i}" for i in range(n_agents)]
        self._shared_inboxes: dict[str, list[str]] = {aid: [] for aid in self.agent_ids}

    @property
    def is_active(self) -> bool:
        """Return True if sharing mode is enabled."""
        return self.config.mode != "off"

    def step(
        self,
        round_num: int,
        memory_policies: dict[str, BaseMemoryPolicy],
    ) -> dict[str, int]:
        """Execute one round of memory publishing and routing across the topology.

        Publishing occurs every ``config.publish_every_n_rounds`` rounds.

        Args:
            round_num: Current 1-indexed simulation round number.
            memory_policies: Dict mapping ``{agent_id: memory_policy_instance}``.

        Returns:
            Dict mapping ``{agent_id: n_messages_received}``.
        """
        counts = {aid: 0 for aid in self.agent_ids}
        if not self.is_active:
            return counts

        # Check publishing cadence
        if round_num % self.config.publish_every_n_rounds != 0:
            return counts

        # 1. Collect published memory context from each agent
        published_artifacts: dict[str, str] = {}
        for aid in self.agent_ids:
            policy = memory_policies.get(aid)
            if policy:
                context = policy.get_context()
                if context.strip():
                    published_artifacts[aid] = self._truncate_artifact(context)

        # 2. Route messages to neighbors according to topology
        for target_id in self.agent_ids:
            neighbors = self.topology.get_neighbors(target_id)
            received_chunks = []

            for sender_id in neighbors:
                if sender_id in published_artifacts:
                    snippet = published_artifacts[sender_id]
                    received_chunks.append(f"[{sender_id}]: {snippet}")

            # Apply consume mode filtering (e.g. top_k or all)
            if self.config.consume_mode == "top_k" and len(received_chunks) > 2:
                received_chunks = received_chunks[:2]

            self._shared_inboxes[target_id] = received_chunks
            counts[target_id] = len(received_chunks)

        logger.debug(
            "Round %d: MemorySharingEngine routed memory updates across '%s' topology.",
            round_num,
            self.topology.topology_type,
        )
        return counts

    def get_shared_context(self, agent_id: str) -> str:
        """Return formatted string of incoming shared memories for *agent_id*.

        Args:
            agent_id: Target agent identifier string.

        Returns:
            Formatted multi-line snippet string to embed in prompt.
        """
        inbox = self._shared_inboxes.get(agent_id, [])
        if not inbox or not self.is_active:
            return ""

        lines = ["=== Shared Peer Memories ==="]
        lines.extend(inbox)
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all agent shared memory inboxes."""
        for aid in self.agent_ids:
            self._shared_inboxes[aid].clear()

    def close(self) -> None:
        """Release engine resources."""
        self.clear()

    def _truncate_artifact(self, text: str) -> str:
        """Truncate published memory text to respect max_artifact_tokens budget."""
        max_chars = self.config.max_artifact_tokens * 4  # ~4 chars per token rule of thumb
        text_clean = text.strip().replace("\n", " ")
        if len(text_clean) > max_chars:
            return text_clean[:max_chars] + "..."
        return text_clean
