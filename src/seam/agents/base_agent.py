"""BaseAgent implementation wrapping prompt construction, LLM completion, and action extraction."""

from __future__ import annotations

import logging
import re
from typing import Any

from seam.agents.decoding import OllamaClient
from seam.orchestration.config_loader import ModelConfig

logger = logging.getLogger(__name__)


class BaseAgent:
    """Represents an individual LLM agent in SEAM.

    Args:
        agent_id: Unique string identifier (e.g. ``"agent_0"``).
        model_config: Configuration for the agent's LLM.
        client: Optional pre-initialised :class:`OllamaClient`. If None, one will be created.
        system_prompt: System prompt / role description for the agent.
    """

    def __init__(
        self,
        agent_id: str,
        model_config: ModelConfig,
        client: OllamaClient | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.config = model_config
        self.client = client or OllamaClient(model_config)
        self.system_prompt = system_prompt or (
            "You are a helpful, rational agent acting in a multi-agent environment."
        )

    def format_prompt(
        self,
        observation: dict[str, Any],
        action_space: list[str],
        memory_context: str = "",
    ) -> str:
        """Construct the prompt sent to the LLM.

        Args:
            observation: Current observation dict from the environment.
            action_space: List of valid action strings.
            memory_context: Optional memory artifact / playbook text.

        Returns:
            Fully assembled prompt string.
        """
        lines = [
            f"=== System ===",
            self.system_prompt,
            "",
            f"=== Agent ID ===",
            self.agent_id,
        ]

        if memory_context.strip():
            lines.extend([
                "",
                "=== Memory & Strategies ===",
                memory_context.strip(),
            ])

        lines.extend([
            "",
            "=== Current Observation ===",
            str(observation),
            "",
            "=== Available Actions ===",
            ", ".join(action_space[:10]) + ("..." if len(action_space) > 10 else ""),
            "",
            "=== Instruction ===",
            "Respond with your chosen action. Your response must clearly contain one of the valid action strings.",
            "Action:",
        ])
        return "\n".join(lines)

    def extract_action(
        self,
        raw_response: str,
        action_space: list[str],
        default_action: str | None = None,
    ) -> str:
        """Extract a valid action from raw LLM output.

        Searches for exact matches or regex patterns corresponding to items in *action_space*.
        If no match is found, returns *default_action* or the first valid action in *action_space*.

        Args:
            raw_response: Raw text output from the LLM.
            action_space: List of acceptable action strings.
            default_action: Fallback action string if extraction fails.

        Returns:
            A valid action string belonging to *action_space*.
        """
        fallback = default_action or (action_space[0] if action_space else "stay")
        cleaned = raw_response.strip()

        # 1. Exact match (case insensitive)
        for act in action_space:
            if cleaned.lower() == act.lower():
                return act

        # 2. Check for action keyword in the text (word boundary match)
        for act in action_space:
            pattern = rf"\b{re.escape(act)}\b"
            if re.search(pattern, cleaned, re.IGNORECASE):
                return act

        # 3. Check for ultimatum bargaining splits like "50 50" if present in action space
        split_match = re.search(r"\b(\d+)\s+(\d+)\b", cleaned)
        if split_match:
            candidate = f"{split_match.group(1)} {split_match.group(2)}"
            if candidate in action_space:
                return candidate

        logger.warning(
            "[%s] Could not extract valid action from response '%s' — using fallback '%s'",
            self.agent_id,
            cleaned,
            fallback,
        )
        return fallback

    def act(
        self,
        observation: dict[str, Any],
        action_space: list[str],
        memory_context: str = "",
        default_action: str | None = None,
    ) -> str:
        """Generate an action given the observation and memory context.

        Args:
            observation: Current observation dict.
            action_space: Available action strings.
            memory_context: Optional memory context.
            default_action: Fallback action if extraction fails or LLM errors out.

        Returns:
            Selected action string.
        """
        prompt = self.format_prompt(observation, action_space, memory_context)
        fallback = default_action or (action_space[0] if action_space else "stay")

        try:
            raw_response, _latency = self.client.complete(prompt)
            return self.extract_action(raw_response, action_space, default_action=fallback)
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] Error during LLM completion: %s — falling back to '%s'", self.agent_id, exc, fallback)
            return fallback
