"""EpisodeRunner orchestrating multi-agent/single-agent environment simulation episodes."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

from seam.agents.decoding import OllamaClient
from seam.agents.population import AgentPopulation
from seam.envs.bargaining_game import BargainingGame
from seam.envs.number_guessing import NumberGuessingGame
from seam.envs.resource_foraging import ResourceForagingGame
from seam.logging.run_logger import RunLogger
from seam.memory.base_memory import BaseMemoryPolicy
from seam.memory.factory import create_memory_policy
from seam.metrics.collapse import (
    compute_action_entropy,
    compute_embedding_similarity,
    compute_memory_length,
    compute_self_bleu,
)
from seam.orchestration.config_loader import ExperimentConfig
from seam.utils.reproducibility import set_seed

from seam.metrics.contamination import (
    compute_contamination_rate,
    compute_poison_adherence,
)
from seam.poisoning.injector import PoisonInjector
from seam.sharing.engine import MemorySharingEngine

logger = logging.getLogger(__name__)


class EpisodeRunner:
    """Coordinates environment step loop, agent population actions, memory updates, and logging.

    Args:
        config: Experiment configuration object.
        seed: Random seed for episode execution.
        client: Optional shared :class:`OllamaClient`.
        base_dir: Parent output directory for run logs (default ``"runs"``).
    """

    def __init__(
        self,
        config: ExperimentConfig,
        seed: int,
        client: OllamaClient | None = None,
        base_dir: str | Path = "runs",
    ) -> None:
        self.config = config
        self.seed = seed
        self.client = client
        self.logger_inst = RunLogger(config=config, seed=seed, base_dir=base_dir)

        # 1. Initialize environment based on config
        env_type = config.env.type.lower().strip()
        if env_type == "resource_foraging":
            self.env = ResourceForagingGame(
                n_agents=config.env.n_agents,
                grid_size=config.env.grid_size,
                episode_length=config.env.episode_length,
                resource_spawn_rate=config.env.resource_spawn_rate,
            )
        elif env_type == "bargaining_game":
            self.env = BargainingGame(
                n_agents=config.env.n_agents,
                episode_length=config.env.episode_length,
            )
        elif env_type == "number_guessing":
            self.env = NumberGuessingGame(
                n_agents=config.env.n_agents,
                episode_length=config.env.episode_length,
            )
        else:
            raise ValueError(f"Unsupported environment type '{config.env.type}'")

        # 2. Initialize Population, Memory Policies, Sharing Engine & Poison Injector
        self.population = AgentPopulation(
            n_agents=config.env.n_agents,
            model_config=config.model,
            client=self.client,
        )
        self.memory_policies: dict[str, BaseMemoryPolicy] = {
            aid: create_memory_policy(config.memory) for aid in self.population.agent_ids
        }
        self.sharing_engine = MemorySharingEngine(
            config=config.sharing,
            n_agents=config.env.n_agents,
            topology_type=config.sharing.topology,
        )
        self.poison_injector = PoisonInjector(
            config=config.poisoning,
            env_type=config.env.type,
        )

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Explicitly release all resources held by this runner.

        Closes the :class:`AgentPopulation` (and its shared OllamaClient
        connection pool), clears memory policies, resets the environment
        state, closes the sharing engine, and forces a CPython GC cycle so
        that freed heap pages are returned to the OS promptly. Call this
        after :meth:`run` returns.
        """
        # 1. Close population → closes shared OllamaClient httpx pool
        self.population.close()
        logger.debug("EpisodeRunner: population closed.")

        # 2. Drop all memory policy objects & close sharing engine
        self.memory_policies.clear()
        self.sharing_engine.close()
        logger.debug("EpisodeRunner: memory policies & sharing engine cleared.")

        # 3. Reset environment internal state
        if hasattr(self.env, "reset"):
            try:
                self.env.reset(seed=0)
            except Exception:  # noqa: BLE001
                pass  # best-effort

        # 4. Force GC — reclaim any cyclic references lingering in response objects
        gc.collect()
        logger.debug("EpisodeRunner: gc.collect() completed.")

    def __enter__(self) -> "EpisodeRunner":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def run(self) -> dict[str, Any]:
        """Execute the full episode.

        Returns:
            Summary dictionary containing final score, cumulative rewards, collapse metrics, and contamination metrics.
        """
        set_seed(self.seed)
        obs = self.env.reset(self.seed)
        action_space = self.env.action_space

        # Inject initial internal poison if configured
        self.poison_injector.inject_initial_memory(self.memory_policies)

        per_agent_actions: dict[str, list[str]] = {aid: [] for aid in self.population.agent_ids}
        per_agent_rewards: dict[str, list[float]] = {aid: [] for aid in self.population.agent_ids}
        per_agent_memories: dict[str, list[str]] = {aid: [] for aid in self.population.agent_ids}

        done = False
        round_num = 0

        while not done:
            round_num += 1

            # Inject channel/gradual poison if active
            self.poison_injector.inject_channel(self.sharing_engine, round_num=round_num)

            # Step sharing engine to process and route memories across topology
            self.sharing_engine.step(round_num=round_num, memory_policies=self.memory_policies)

            # Get combined memory contexts (local policy + shared peer context)
            memory_contexts = {}
            for aid, policy in self.memory_policies.items():
                local_ctx = policy.get_context()
                shared_ctx = self.sharing_engine.get_shared_context(aid)
                if shared_ctx:
                    memory_contexts[aid] = f"{local_ctx}\n\n{shared_ctx}" if local_ctx else shared_ctx
                else:
                    memory_contexts[aid] = local_ctx

            # Collect actions from all agents
            actions = self.population.act_all(
                observations=obs,
                action_space=action_space,
                memory_contexts=memory_contexts,
            )

            # Step environment
            step_result = self.env.step(actions)
            next_obs = step_result["observations"]
            rewards = step_result["rewards"]
            done = step_result["done"]
            info = step_result["info"]

            # Update memories & record logs
            for aid in self.population.agent_ids:
                act = actions[aid]
                rew = rewards.get(aid, 0.0)
                ob = obs.get(aid, {})

                experience = {"observation": ob, "action": act, "reward": rew}
                updated_mem = self.memory_policies[aid].update(experience, client=self.client)

                per_agent_actions[aid].append(act)
                per_agent_rewards[aid].append(rew)
                per_agent_memories[aid].append(updated_mem)

                self.logger_inst.log_step(
                    round_num=round_num,
                    agent_id=aid,
                    observation=ob,
                    prompt="",  # BaseAgent formats prompts internally
                    raw_response=act,
                    action=act,
                    reward=rew,
                    memory_state=updated_mem,
                    info=info,
                )

            obs = next_obs

        # Compute episode evaluation metrics
        ground_truth_score = self.env.get_ground_truth_score()
        cumulative_rewards = {aid: sum(rews) for aid, rews in per_agent_rewards.items()}

        per_agent_self_bleu = {
            aid: compute_self_bleu(mems) for aid, mems in per_agent_memories.items()
        }
        per_agent_action_entropy = {
            aid: compute_action_entropy(acts) for aid, acts in per_agent_actions.items()
        }
        per_agent_memory_lengths = {
            aid: compute_memory_length(mems) for aid, mems in per_agent_memories.items()
        }

        # Contamination metrics
        poison_payload = self.poison_injector.poison_payload
        poison_keywords = [k for k in poison_payload.split() if len(k) > 3][:3]
        peer_contamination_rate = compute_contamination_rate(
            per_agent_memories=per_agent_memories,
            poison_keywords=poison_keywords,
            seed_agent_id=self.config.poisoning.poison_agent_id,
        ) if self.poison_injector.is_active else 0.0

        per_agent_poison_adherence = {
            aid: compute_poison_adherence(acts, target_pattern=poison_payload)
            for aid, acts in per_agent_actions.items()
        } if self.poison_injector.is_active else {aid: 0.0 for aid in self.population.agent_ids}

        summary = {
            "run_id": self.logger_inst.run_id,
            "seed": self.seed,
            "rounds_played": round_num,
            "final_score": ground_truth_score,
            "cumulative_rewards": cumulative_rewards,
            "mean_self_bleu": sum(per_agent_self_bleu.values()) / len(per_agent_self_bleu),
            "per_agent_self_bleu": per_agent_self_bleu,
            "per_agent_action_entropy": per_agent_action_entropy,
            "per_agent_memory_lengths": per_agent_memory_lengths,
            "peer_contamination_rate": peer_contamination_rate,
            "per_agent_poison_adherence": per_agent_poison_adherence,
        }

        self.logger_inst.log_episode_end(final_score=ground_truth_score, summary_info=summary)
        return summary
