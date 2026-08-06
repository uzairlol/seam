"""Unit tests for AgentPopulation multi-agent management."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from seam.agents.population import AgentPopulation
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


def test_agent_population_initialization(model_config: ModelConfig) -> None:
    """Test population initializes specified number of agents with IDs."""
    mock_client = MagicMock()
    pop = AgentPopulation(n_agents=4, model_config=model_config, client=mock_client)

    assert pop.n_agents == 4
    assert pop.agent_ids == ["agent_0", "agent_1", "agent_2", "agent_3"]
    assert pop.get_agent("agent_0").agent_id == "agent_0"


def test_agent_population_get_invalid_agent(model_config: ModelConfig) -> None:
    """Test KeyError raised for nonexistent agent ID."""
    mock_client = MagicMock()
    pop = AgentPopulation(n_agents=2, model_config=model_config, client=mock_client)

    with pytest.raises(KeyError):
        pop.get_agent("agent_999")


def test_agent_population_act_all(model_config: ModelConfig) -> None:
    """Test act_all collects actions from all agents."""
    mock_client = MagicMock()
    mock_client.complete.return_value = ("Action: harvest", 10)
    pop = AgentPopulation(n_agents=3, model_config=model_config, client=mock_client)

    observations = {
        "agent_0": {"pos": [0, 0]},
        "agent_1": {"pos": [1, 1]},
        "agent_2": {"pos": [2, 2]},
    }
    action_space = ["up", "down", "harvest", "stay"]
    memory_contexts = {
        "agent_0": "Strategy A",
        "agent_1": "Strategy B",
    }

    actions = pop.act_all(
        observations=observations,
        action_space=action_space,
        memory_contexts=memory_contexts,
    )

    assert len(actions) == 3
    assert actions["agent_0"] == "harvest"
    assert actions["agent_1"] == "harvest"
    assert actions["agent_2"] == "harvest"
    assert mock_client.complete.call_count == 3
