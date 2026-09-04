# Task Scenarios

All three environments implement the `BaseEnv` contract: action space, agent count, seeded reset, step transition, objective score, and rendering.

## Resource foraging

**Implementation:** `src/seam/envs/resource_foraging.py`

- Default world: 10x10 grid.
- Agents begin at distinct random positions.
- Actions: `stay`, `north`, `south`, `east`, `west`, `harvest`.
- Movement is simultaneous and clipped at grid boundaries.
- If agents collide, their harvest rewards are nullified for that cell.
- A single non-colliding harvester receives `1.0` when resources are present.
- Empty cells may spawn resources each round.
- Default episode length in the environment is 50 rounds; the experiment scripts construct 20-round runs.
- Ground-truth score: harvested resources divided by total spawned resources, with a zero-safe fallback.

This is the primary test of whether memory helps agents discover and preserve spatially useful behavior.

## Bargaining game

**Implementation:** `src/seam/envs/bargaining_game.py`

- A proposer and responder are selected each round.
- The proposer emits two integer shares summing to the pie size, normally 100.
- The responder emits `accept` or `reject`.
- Accepted deals award the proposed shares; rejected deals award zero.
- The current `get_ground_truth_score()` is a backward-compatible alias for `get_fairness_score()`.
- Fairness score is `1 / (1 + mean absolute deviation of the proposer's share from 50)` across accepted deals.
- This is a fairness metric, not total-welfare or efficiency score. A 99/1 accepted deal has full welfare but a low fairness score.

Manuscripts should call this metric fairness unless explicitly discussing the compatibility method name.

## Number guessing

**Implementation:** `src/seam/envs/number_guessing.py`

- One secret integer is sampled from the configured inclusive range.
- Every agent submits an integer guess each round.
- Each agent receives `higher`, `lower`, or `correct` feedback.
- If any agent guesses correctly, all agents receive `1.0` and the episode ends.
- If the episode reaches its maximum length, the episode ends unsolved.
- Ground-truth score: `1 / rounds_to_solve`, or `0.0` if unsolved.

This is the simplest validation environment for action parsing, feedback use, and collapse/poison metrics.

## Cross-scenario comparability

Scores are normalized or bounded within each environment, but their semantics differ. Do not pool raw scores across scenarios without identifying the environment. Compare policies within a scenario first, then discuss cross-scenario patterns cautiously.
