"""Tests for ResourceForagingGame scoring, determinism, and mechanics."""

from __future__ import annotations

import random

import pytest

from seam.envs.resource_foraging import ResourceForagingGame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_random_episode(seed: int, episode_length: int = 50) -> dict:
    """Run a full episode with uniformly random actions; return final result."""
    env = ResourceForagingGame(episode_length=episode_length)
    rng = random.Random(seed)
    obs = env.reset(seed)
    agent_ids = list(obs.keys())
    result = {}
    for _ in range(episode_length):
        actions = {aid: rng.choice(env.action_space) for aid in agent_ids}
        result = env.step(actions)
    return result


# ---------------------------------------------------------------------------
# Test 1 — Determinism: same seed → identical episode
# ---------------------------------------------------------------------------

def test_same_seed_same_episode() -> None:
    """Two runs with the same seed must produce identical reward sequences."""
    seed = 42

    def collect_rewards(seed: int) -> list[dict[str, float]]:
        env = ResourceForagingGame(episode_length=10)
        rng = random.Random(seed)
        obs = env.reset(seed)
        agent_ids = list(obs.keys())
        history: list[dict[str, float]] = []
        for _ in range(10):
            actions = {aid: rng.choice(env.action_space) for aid in agent_ids}
            step_result = env.step(actions)
            history.append(step_result["rewards"])
        return history

    run_a = collect_rewards(seed)
    run_b = collect_rewards(seed)
    assert run_a == run_b, "Same seed produced different reward histories"


# ---------------------------------------------------------------------------
# Test 2 — Different seeds produce different episodes
# ---------------------------------------------------------------------------

def test_different_seeds_different_episodes() -> None:
    """Sanity check: different seeds should (almost certainly) differ."""
    def collect_rewards(seed: int) -> list[dict[str, float]]:
        env = ResourceForagingGame(episode_length=10)
        rng = random.Random(seed)
        obs = env.reset(seed)
        agent_ids = list(obs.keys())
        history = []
        for _ in range(10):
            actions = {aid: rng.choice(env.action_space) for aid in agent_ids}
            step_result = env.step(actions)
            history.append(step_result["rewards"])
        return history

    assert collect_rewards(1) != collect_rewards(2)


# ---------------------------------------------------------------------------
# Test 3 — Collision gives 0 reward
# ---------------------------------------------------------------------------

def test_collision_gives_zero_reward() -> None:
    """Two agents on the same cell both get 0 reward, even if they harvest."""
    env = ResourceForagingGame(n_agents=2, grid_size=5, episode_length=5)
    env.reset(seed=0)

    # Force both agents onto the same cell
    env._positions = {"agent_0": [2, 2], "agent_1": [2, 2]}
    # Place a resource there
    env._grid[2, 2] = 3

    result = env.step({"agent_0": "harvest", "agent_1": "harvest"})
    rewards = result["rewards"]
    assert rewards["agent_0"] == 0.0, "Colliding agent_0 should get 0 reward"
    assert rewards["agent_1"] == 0.0, "Colliding agent_1 should get 0 reward"


# ---------------------------------------------------------------------------
# Test 4 — Harvest on empty cell gives 0 reward
# ---------------------------------------------------------------------------

def test_harvest_empty_cell_gives_zero_reward() -> None:
    """Harvesting a cell with 0 resources yields 0 reward."""
    env = ResourceForagingGame(n_agents=2, grid_size=5, episode_length=5)
    env.reset(seed=0)

    # Clear all resources; place agents on distinct empty cells
    env._grid[:] = 0
    env._positions = {"agent_0": [0, 0], "agent_1": [4, 4]}

    result = env.step({"agent_0": "harvest", "agent_1": "harvest"})
    assert result["rewards"]["agent_0"] == 0.0
    assert result["rewards"]["agent_1"] == 0.0


# ---------------------------------------------------------------------------
# Test 5 — get_ground_truth_score() is in [0.0, 1.0]
# ---------------------------------------------------------------------------

def test_ground_truth_score_in_range() -> None:
    """After any episode, get_ground_truth_score() must be in [0.0, 1.0]."""
    env = ResourceForagingGame(episode_length=50)
    rng = random.Random(99)
    obs = env.reset(99)
    agent_ids = list(obs.keys())
    for _ in range(50):
        actions = {aid: rng.choice(env.action_space) for aid in agent_ids}
        env.step(actions)

    score = env.get_ground_truth_score()
    assert 0.0 <= score <= 1.0, f"Score {score} is outside [0.0, 1.0]"


# ---------------------------------------------------------------------------
# Test 6 — Episode terminates after episode_length rounds
# ---------------------------------------------------------------------------

def test_episode_terminates_after_episode_length() -> None:
    """The 'done' flag must be True exactly when episode_length rounds pass."""
    episode_length = 10
    env = ResourceForagingGame(episode_length=episode_length)
    rng = random.Random(7)
    obs = env.reset(7)
    agent_ids = list(obs.keys())

    done = False
    rounds_played = 0
    while not done:
        actions = {aid: rng.choice(env.action_space) for aid in agent_ids}
        result = env.step(actions)
        done = result["done"]
        rounds_played += 1
        assert rounds_played <= episode_length, "Episode ran longer than episode_length"

    assert rounds_played == episode_length, (
        f"Episode ended after {rounds_played} rounds, expected {episode_length}"
    )


# ---------------------------------------------------------------------------
# Test 7 — step() raises after episode is done
# ---------------------------------------------------------------------------

def test_step_after_done_raises() -> None:
    """Calling step() after the episode finishes should raise RuntimeError."""
    env = ResourceForagingGame(episode_length=3)
    rng = random.Random(0)
    obs = env.reset(0)
    agent_ids = list(obs.keys())
    for _ in range(3):
        env.step({aid: rng.choice(env.action_space) for aid in agent_ids})

    with pytest.raises(RuntimeError):
        env.step({aid: "stay" for aid in agent_ids})


# ---------------------------------------------------------------------------
# Test 8 — Successful single-agent harvest gives reward 1.0
# ---------------------------------------------------------------------------

def test_successful_harvest_gives_reward_one() -> None:
    """A lone agent harvesting a cell with resources gets reward 1.0."""
    env = ResourceForagingGame(n_agents=2, grid_size=5, episode_length=5)
    env.reset(seed=0)

    # Clear grid, put agents on different cells, put resource under agent_0
    env._grid[:] = 0
    env._positions = {"agent_0": [1, 1], "agent_1": [3, 3]}
    env._grid[1, 1] = 2

    result = env.step({"agent_0": "harvest", "agent_1": "stay"})
    assert result["rewards"]["agent_0"] == 1.0
    assert result["rewards"]["agent_1"] == 0.0


# ---------------------------------------------------------------------------
# BargainingGame tests
# ---------------------------------------------------------------------------

from seam.envs.bargaining_game import BargainingGame
from seam.envs.number_guessing import NumberGuessingGame


def test_bargaining_game_mechanics_and_scoring() -> None:
    """Test BargainingGame proposal parsing, deal acceptance/rejection, and score."""
    env = BargainingGame(n_agents=2, episode_length=5, pie_size=100)
    obs = env.reset(seed=42)
    assert len(obs) == 2
    assert "agent_0" in obs and "agent_1" in obs

    # Step 1: Valid accepted deal (50 50)
    result = env.step({"agent_0": "50 50", "agent_1": "accept"})
    assert not result["done"]
    info = result["info"]
    proposer = info["deal"]["proposer"]
    responder = info["deal"]["responder"]
    assert result["rewards"][proposer] == 50.0
    assert result["rewards"][responder] == 50.0

    # Step 2: Rejected deal
    result = env.step({"agent_0": "80 20", "agent_1": "reject"})
    info = result["info"]
    proposer = info["deal"]["proposer"]
    responder = info["deal"]["responder"]
    assert result["rewards"][proposer] == 0.0
    assert result["rewards"][responder] == 0.0

    score = env.get_ground_truth_score()
    assert 0.0 < score <= 1.0


def test_bargaining_game_invalid_proposal_fallback() -> None:
    """Invalid proposer string falls back to 50/50 split."""
    env = BargainingGame(n_agents=2, episode_length=2, pie_size=100)
    env.reset(seed=10)
    result = env.step({"agent_0": "invalid_proposal", "agent_1": "accept"})
    info = result["info"]
    assert info["deal"]["proposed_split"] == [50, 50]


def test_bargaining_game_no_accepted_deals_score_zero() -> None:
    """If no deals are accepted, get_ground_truth_score() must return 0.0."""
    env = BargainingGame(n_agents=2, episode_length=3, pie_size=100)
    env.reset(seed=77)
    for _ in range(3):
        env.step({"agent_0": "90 10", "agent_1": "reject"})
    assert env.get_ground_truth_score() == 0.0


# ---------------------------------------------------------------------------
# NumberGuessingGame tests
# ---------------------------------------------------------------------------

def test_number_guessing_game_mechanics() -> None:
    """Test NumberGuessingGame feedback, solver reward, and ground truth score."""
    env = NumberGuessingGame(n_agents=2, episode_length=10, secret_min=1, secret_max=100)
    obs = env.reset(seed=123)
    secret = env._secret

    # Submit wrong guesses first
    action = "1" if secret > 1 else "100"
    result = env.step({"agent_0": action, "agent_1": action})
    assert not result["done"]

    # Submit correct guess
    result = env.step({"agent_0": str(secret), "agent_1": str(secret)})
    assert result["done"]
    assert result["rewards"]["agent_0"] == 1.0
    assert result["rewards"]["agent_1"] == 1.0

    score = env.get_ground_truth_score()
    assert score == 0.5  # solved in 2 rounds -> 1/2 = 0.5

