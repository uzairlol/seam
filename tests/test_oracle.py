"""Unit tests for deterministic oracle reference policies."""

from __future__ import annotations

from seam.analysis.oracle import oracle_policy, oracle_rollout
from seam.envs.bargaining_game import BargainingGame
from seam.envs.number_guessing import NumberGuessingGame
from seam.envs.resource_foraging import ResourceForagingGame


def test_foraging_oracle_turns_toward_visible_resource() -> None:
    obs = {
        "my_position": [2, 2],
        "visible_resources": [{"row": 5, "col": 2, "quantity": 1}],
    }
    assert oracle_policy("resource_foraging", obs, []) == "south"


def test_foraging_oracle_harvests_when_on_resource() -> None:
    obs = {
        "my_position": [2, 2],
        "visible_resources": [{"row": 2, "col": 2, "quantity": 3}],
    }
    assert oracle_policy("resource_foraging", obs, []) == "harvest"


def test_guessing_oracle_binary_searches() -> None:
    action_space = [str(i) for i in range(1, 101)]
    obs = {"history": [(50, "higher")]}
    assert oracle_policy("number_guessing", obs, action_space) == "75"
    obs = {"history": [(50, "lower")]}
    assert oracle_policy("number_guessing", obs, action_space) == "25"


def test_bargaining_oracle_proposes_fair_fifty_fifty() -> None:
    obs = {"role": "proposer", "pie_size": 100}
    assert oracle_policy("bargaining_game", obs, []) == "50 50"


def test_oracle_rollout_foraging_deterministic_and_bounded() -> None:
    env = ResourceForagingGame(n_agents=6, episode_length=50)
    score_a = oracle_rollout("resource_foraging", env, seed=42)
    score_b = oracle_rollout("resource_foraging", env, seed=42)
    assert score_a == score_b
    assert 0.0 <= score_a <= 1.0


def test_oracle_rollout_bargaining_reaches_fairness_ceiling() -> None:
    env = BargainingGame(n_agents=2, episode_length=20, pie_size=100)
    score = oracle_rollout("bargaining_game", env, seed=7)
    assert score == 1.0


def test_oracle_rollout_guessing_solves_expected_rounds() -> None:
    env = NumberGuessingGame(n_agents=1, episode_length=30, secret_min=1, secret_max=100)
    score = oracle_rollout("number_guessing", env, seed=11)
    assert score > 0.0
    assert score >= 1.0 / 8  # binary search never needs more than ~7 rounds
