"""Deterministic optimal reference (oracle) policies for SEAM task environments.

An oracle rollout runs a rule-based policy on a fresh environment instance
seeded identically to the episode under evaluation.  Because every SEAM
environment is fully deterministic given its seed, the oracle score is a
reproducible upper reference per ``(env_type, seed)`` pair.  Comparing a run's
final score against the oracle yields an *efficacy gap*: how much of the
theoretical headroom above the ``no_memory`` baseline a policy actually
recovered.
"""

from __future__ import annotations

from typing import Any

from seam.envs.base_env import BaseEnv


def oracle_policy(env_type: str, obs: dict[str, Any], action_space: list[str]) -> str:
    """Choose the next deterministic optimal action for one agent.

    All policies are rule-based and require no LLM judge.

    Args:
        env_type: Environment type string (``resource_foraging``,
            ``number_guessing``, ``bargaining_game``).
        obs: The agent's observation dict.
        action_space: List of valid action strings for the environment.

    Returns:
        A deterministic reference action string.
    """
    if env_type == "resource_foraging":
        return _foraging_policy(obs)
    if env_type == "number_guessing":
        return _guessing_policy(obs, action_space)
    if env_type == "bargaining_game":
        return _bargaining_policy(obs)
    raise ValueError(f"Unsupported environment type '{env_type}' for oracle rollout")


def oracle_rollout(env_type: str, env: BaseEnv, seed: int) -> float:
    """Roll out the oracle reference policy on a fresh environment state.

    The environment is reset with *seed* so its dynamics exactly mirror the
    episode being evaluated, then stepped with oracle actions until done.

    Args:
        env_type: Environment type string.
        env: An environment instance (its internal state is reset here).
        seed: Seed used for the comparable episode.

    Returns:
        The oracle's ground-truth score for that (env_type, seed) instance.
    """
    action_space = env.action_space
    obs = env.reset(seed)
    done = False
    while not done:
        actions = {aid: oracle_policy(env_type, obs[aid], action_space) for aid in obs}
        result = env.step(actions)
        obs = result["observations"]
        done = bool(result["done"])
    return float(env.get_ground_truth_score())


def _foraging_policy(obs: dict[str, Any]) -> str:
    """Greedily move toward the nearest visible resource and harvest on arrival."""
    row, col = obs["my_position"]
    visible = obs.get("visible_resources", [])
    if visible:
        target = min(
            visible,
            key=lambda res: abs(res["row"] - row) + abs(res["col"] - col),
        )
        t_row, t_col = target["row"], target["col"]
        if t_row == row and t_col == col:
            return "harvest"
        if t_row < row:
            return "north"
        if t_row > row:
            return "south"
        if t_col < col:
            return "west"
        return "east"
    return "stay"


def _guessing_policy(obs: dict[str, Any], action_space: list[str]) -> str:
    """Binary search over the action space using the agent's own feedback history."""
    ints = [int(a) for a in action_space if str(a).lstrip("-").isdigit()] or [1, 100]
    lo, hi = min(ints), max(ints)
    for guess, feedback in obs.get("history", []):
        g = int(guess)
        if feedback == "higher":
            lo = max(lo, g + 1)
        elif feedback == "lower":
            hi = min(hi, g - 1)
    mid = (lo + hi) // 2
    if mid < lo:
        mid = lo
    return str(mid)


def _bargaining_policy(obs: dict[str, Any]) -> str:
    """Propose the Nash 50/50 split; accept any proposal; always play for observers."""
    role = obs.get("role", "observer")
    if role == "proposer":
        pie = int(obs.get("pie_size", 100))
        half = pie // 2
        return f"{half} {pie - half}"
    if role == "responder":
        return "accept"
    return "wait"
