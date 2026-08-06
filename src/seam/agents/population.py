"""AgentPopulation manager for handling multiple LLM agents in simulation episodes."""

from __future__ import annotations

import logging
from typing import Any

from seam.agents.base_agent import BaseAgent
from seam.agents.decoding import OllamaClient
from seam.orchestration.config_loader import ModelConfig

logger = logging.getLogger(__name__)


class AgentPopulation:
    """Manages an ensemble of LLM agents participating in an environment.

    Args:
        n_agents: Number of agents in the population.
        model_config: Configuration for all agents' LLMs.
        client: Optional shared :class:`OllamaClient`.
        system_prompts: Optional mapping of ``{agent_id: system_prompt}``.
    """

    def __init__(
        self,
        n_agents: int,
        model_config: ModelConfig,
        client: OllamaClient | None = None,
        system_prompts: dict[str, str] | None = None,
    ) -> None:
        self.n_agents = n_agents
        self.config = model_config
        self.shared_client = client or OllamaClient(model_config)

        prompts = system_prompts or {}
        self.agents: dict[str, BaseAgent] = {}
        for i in range(n_agents):
            aid = f"agent_{i}"
            sys_prompt = prompts.get(aid)
            self.agents[aid] = BaseAgent(
                agent_id=aid,
                model_config=model_config,
                client=self.shared_client,
                system_prompt=sys_prompt,
            )

    @property
    def agent_ids(self) -> list[str]:
        """Return the list of agent IDs."""
        return list(self.agents.keys())

    def get_agent(self, agent_id: str) -> BaseAgent:
        """Retrieve an agent by its ID.

        Args:
            agent_id: The ID of the desired agent.

        Returns:
            The requested :class:`BaseAgent`.

        Raises:
            KeyError: If agent_id is not in the population.
        """
        if agent_id not in self.agents:
            raise KeyError(f"Agent '{agent_id}' not found in population.")
        return self.agents[agent_id]

    def act_all(
        self,
        observations: dict[str, dict[str, Any]],
        action_space: list[str],
        memory_contexts: dict[str, str] | None = None,
        default_actions: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Collect actions from all agents in parallel or sequentially.

        Args:
            observations: Dict of ``{agent_id: observation_dict}``.
            action_space: List of valid action strings.
            memory_contexts: Optional dict of ``{agent_id: memory_prompt}``.
            default_actions: Optional dict of fallback actions per agent.

        Returns:
            Dict of ``{agent_id: chosen_action}``.
        """
        memories = memory_contexts or {}
        defaults = default_actions or {}
        actions: dict[str, str] = {}

        for aid, agent in self.agents.items():
            obs = observations.get(aid, {})
            mem_ctx = memories.get(aid, "")
            default_act = defaults.get(aid)
            action = agent.act(
                observation=obs,
                action_space=action_space,
                memory_context=mem_ctx,
                default_action=default_act,
            )
            actions[aid] = action

        return actions
