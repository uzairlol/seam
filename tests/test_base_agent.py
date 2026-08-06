"""Unit tests for BaseAgent prompt formatting, action extraction, and fallback behaviour."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from seam.agents.base_agent import BaseAgent
from seam.orchestration.config_loader import ModelConfig


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        model_name="qwen2.5:7b-instruct",
        base_url="http://localhost:11434",
        temperature=0.0,
        seed=42,
        max_tokens=256,
        top_p=1.0,
        request_timeout=10,
        retry_attempts=3,
    )


def test_base_agent_format_prompt(model_config: ModelConfig) -> None:
    """Test format_prompt output structure with and without memory context."""
    mock_client = MagicMock()
    agent = BaseAgent(agent_id="agent_0", model_config=model_config, client=mock_client)

    obs = {"position": [1, 2], "resources": 3}
    action_space = ["up", "down", "harvest", "stay"]

    prompt_no_mem = agent.format_prompt(obs, action_space)
    assert "agent_0" in prompt_no_mem
    assert "Current Observation" in prompt_no_mem
    assert "harvest" in prompt_no_mem
    assert "Memory & Strategies" not in prompt_no_mem

    prompt_with_mem = agent.format_prompt(obs, action_space, memory_context="Always harvest when on resource!")
    assert "Memory & Strategies" in prompt_with_mem
    assert "Always harvest when on resource!" in prompt_with_mem


def test_base_agent_extract_action_exact_and_keyword(model_config: ModelConfig) -> None:
    """Test action extraction for exact matches and keyword occurrences."""
    mock_client = MagicMock()
    agent = BaseAgent(agent_id="agent_0", model_config=model_config, client=mock_client)
    action_space = ["up", "down", "left", "right", "harvest", "stay"]

    assert agent.extract_action("harvest", action_space) == "harvest"
    assert agent.extract_action("I decide to move UP towards the center.", action_space) == "up"
    assert agent.extract_action("Let us STAY here.", action_space) == "stay"


def test_base_agent_extract_action_bargaining(model_config: ModelConfig) -> None:
    """Test action extraction for split actions in bargaining game."""
    mock_client = MagicMock()
    agent = BaseAgent(agent_id="agent_0", model_config=model_config, client=mock_client)
    action_space = ["accept", "reject", "50 50", "60 40"]

    assert agent.extract_action("I propose 50 50 for a fair deal.", action_space) == "50 50"
    assert agent.extract_action("I ACCEPT this offer.", action_space) == "accept"


def test_base_agent_extract_action_fallback(model_config: ModelConfig) -> None:
    """Test default fallback action when no valid action keyword is present."""
    mock_client = MagicMock()
    agent = BaseAgent(agent_id="agent_0", model_config=model_config, client=mock_client)
    action_space = ["up", "down", "harvest", "stay"]

    extracted = agent.extract_action("Grounded explanation without valid action words.", action_space, default_action="stay")
    assert extracted == "stay"


def test_base_agent_act_success(model_config: ModelConfig) -> None:
    """Test act method calls completion and extracts valid action."""
    mock_client = MagicMock()
    mock_client.complete.return_value = ("My choice is harvest.", 50)
    agent = BaseAgent(agent_id="agent_0", model_config=model_config, client=mock_client)

    action = agent.act(
        observation={"pos": [0, 0]},
        action_space=["up", "harvest", "stay"],
    )
    assert action == "harvest"
    mock_client.complete.assert_called_once()


def test_base_agent_act_exception_fallback(model_config: ModelConfig) -> None:
    """Test act returns fallback action when Ollama completion raises an exception."""
    mock_client = MagicMock()
    mock_client.complete.side_effect = Exception("Ollama server down")
    agent = BaseAgent(agent_id="agent_0", model_config=model_config, client=mock_client)

    action = agent.act(
        observation={"pos": [0, 0]},
        action_space=["up", "harvest", "stay"],
        default_action="stay",
    )
    assert action == "stay"
