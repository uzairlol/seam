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

A configuration knob worth documenting for the paper is `rich_cell_yield`, a non-replenishing stock placed in the exact centre cell at reset. When it is greater than zero, that cell receives `rich_cell_yield` extra resource units, but once an agent harvests the cell to zero it joins `_spent_rich_cells` and never respawns, even on cells that would otherwise spawn each round. This models precisely the "locally correct, non-transferable experience" mechanism in the literature map: a strategy (stand on the centre cell and harvest) pays off early and then becomes worthless because the resource is exhaustible. The knob is disabled (0) in every shipped config and in all current run artifacts, so it is currently unused evidence; it exists as a ready-made non-stationarity condition for a follow-up experiment, not as part of the executed grid.

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

Three implementation details matter for the methods section. First, every round all non-participating agents observe with role `observer` and are expected to emit the action `wait`; the runner short-circuits their decision (it preselects `wait` as a default action) and, crucially, does not feed their zero-reward `wait` experience into memory, so an observer's policy only ingests shared peer context on rounds when it is not playing. Second, the proposer's split is parsed with a strict `^\s*(\d+)\s+(\d+)\s*$` regex and validated to sum exactly to the pie size with non-negative parts; anything invalid — an out-of-budget proposal, a malformed string, or a raw LLM response the extraction layer passed through unchanged — falls back to a 50/50 split rather than crashing the episode. Because `BaseAgent.extract_action()` falls back to the first action in the action space (`accept` for this environment) when no proposer-style split is found, the practical effect is that a proposer whose response fails every extraction rule ends up proposing 50/50. Third, the episode terminates when `round >= episode_length` even if negotiations deadlock, so fairness 0.0 can mean "no deals accepted" rather than any malfunction.

## Number guessing

**Implementation:** `src/seam/envs/number_guessing.py`

- One secret integer is sampled from the configured inclusive range.
- Every agent submits an integer guess each round.
- Each agent receives `higher`, `lower`, or `correct` feedback.
- If any agent guesses correctly, all agents receive `1.0` and the episode ends.
- If the episode reaches its maximum length, the episode ends unsolved.
- Ground-truth score: `1 / rounds_to_solve`, or `0.0` if unsolved.

This is the simplest validation environment for action parsing, feedback use, and collapse/poison metrics.

A parser detail matters for interpreting adherence and collapse numbers: the environment coerces a non-integer guess to `secret_min` before comparison, and then compares the integer directly against the secret, so an out-of-range integer still receives proper `higher`/`lower` feedback rather than being clamped into a canonical range decision. The poison payload for this environment targets the fixed guess `"100"`, and because every round is scored by all agents jointly (any correct guess ends the episode with a reward of 1.0 for every agent), the environment has no per-agent competitive structure; it is best understood as a coordination-speed test.

## Cross-scenario comparability

Scores are normalized or bounded within each environment, but their semantics differ. Do not pool raw scores across scenarios without identifying the environment. Compare policies within a scenario first, then discuss cross-scenario patterns cautiously.

## Environment contract in practice

Each reset returns keys such as `agent_0`, `agent_1`, and so on, matching the population and sharing-engine IDs. Each step returns four objects: a per-agent observation mapping, a per-agent reward mapping, a boolean `done`, and an `info` dictionary. The `info` dictionary is diagnostic rather than an action input; examples include feedbacks and secret visibility at episode termination for number guessing, deal information for bargaining, and collision/spawn totals for foraging.

## Determinism boundary

The environments use seeded NumPy generators for their task state, while the project-level reproducibility helper seeds Python's `random` module and NumPy's global RNG. Model responses can still affect a run unless the Ollama model, decoding options, and server behavior are held constant. Therefore, “same seed” means deterministic task initialization and project RNG setup, not a guarantee of identical output across changed model/runtime configurations.
