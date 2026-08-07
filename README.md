# SEAM — Shared Evolving Agent Memory

> *Does sharing memory help or hurt self-evolving LLM agents? A controlled multi-agent study of collapse and contamination.*

---

## What this project is about

Most research on self-evolving LLM agents studies a single agent learning in isolation. When that agent updates its memory after each interaction, it gets better — or it collapses into a repetitive loop of over-compressed, stale strategies. This collapse phenomenon has been named the **Echo Trap** (RAGEN, 2025) in RL-based evolution and **context collapse** (ACE, ICLR 2026) in memory-based evolution. Both papers found it independently from different directions, which suggests it is a fundamental property of any self-referential learning loop — not an artefact of a particular mechanism.

The poisoning side of this literature (OEP, 2026; Zombie Agents, 2026) adds a second problem: agents that trust their own reflections as ground truth are vulnerable to experiences that are *locally correct* but *non-transferable* — lessons that worked in a narrow context and get over-generalised into persistent, harmful rules.

**Neither problem has been studied in a multi-agent setting.** Every existing paper either runs one agent or, if it runs many, does not measure what happens when their memories interact.

This project fills that gap with a clean, controlled experiment:

- **4–6 agents**, each running a small local LLM (Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct via Ollama)
- **3 memory mechanisms** compared side by side: naive overwrite, raw trajectory buffer, and structured incremental update (ACE-style Generate→Reflect→Curate playbook)
- **A shared broadcast channel** through which agents periodically publish compressed memory artifacts and consume each other's
- **A poisoning condition** where one agent is seeded with a locally-correct but non-transferable lesson, and we track whether and how fast it spreads to the rest of the population

The task environment is deliberately simple — a shared resource foraging grid — so that "did this memory update help or hurt" is answerable from the environment's reward signal alone, without an LLM judge.

---

## The core research question

> When multiple self-evolving agents share memory through a common channel, does the exchange accelerate collapse, amplify contamination, or actually stabilise the population?

This sits at the intersection of two open gaps in the 2025–2026 literature:

1. The connection between the Echo Trap (RL self-evolution) and context collapse (memory self-evolution) as the same underlying failure mode seen from two directions — nobody has written the paper connecting them.
2. Multi-agent memory poisoning via honest peer experience — OEP and Zombie Agents both study one agent poisoning itself or being attacked by a static adversarial document. Nobody has asked what happens when the "poison" is just another self-evolving agent's own non-transferable lesson propagating organically through a shared channel.

---

## What is being built

### Environments
Three task environments with objective, deterministic scoring (no LLM judge needed):
- **Resource Foraging** *(primary)*: shared 10×10 grid, agents harvest resources over 50 rounds
- **Bargaining Game** *(secondary)*: repeated negotiation with a computable Nash equilibrium
- **Number Guessing** *(validation)*: simplest possible environment for metric calibration

### Memory policies (ablation)
| Policy | Mechanism | Expected behaviour |
|--------|-----------|-------------------|
| Naive Overwrite | Full LLM rewrite of memory each round | Fast collapse via brevity bias |
| Raw Trajectory Buffer | Sliding window of raw `(state, action, reward)` tuples | Slower collapse, noisier |
| Structured Incremental Update | ACE-style Generate→Reflect→Curate playbook | Most collapse-resistant; explicit deprecation mechanism |

### Shared channel
- Agents publish a compressed memory artifact every N rounds
- Broadcast pool: all agents consume all available artifacts (sharing ON)
- Baseline: same setup with sharing OFF
- Ablation: selective consumption (highest embedding similarity only)

### Poisoning condition
One agent is seeded with a plausible but non-transferable lesson at round 0. We then measure:
- Does the lesson persist in Agent 0's memory across rounds?
- Does it appear in other agents' memories after the shared-channel exchange?
- Does it measurably degrade their task performance?

### Experimental matrix
3 memory policies × 2 sharing conditions × 2 poisoning conditions × 5–10 seeds = **60–120 runs**

---

## What is being measured

**Collapse metrics:**
- Self-BLEU between successive memory states (lexical repetition)
- Embedding cosine similarity between successive memory states
- Action entropy over a rolling window (behavioural diversity)
- Memory length trajectory (brevity bias proxy)

**Contamination metrics:**
- Poison presence score per agent per round (embedding similarity to the poison lesson)
- Time-to-propagation (first round where another agent crosses the contamination threshold)
- Propagation fraction (what share of agents are contaminated by round T)
- Performance degradation attributable to the contaminated lesson

**Performance metrics:**
- Cumulative reward per agent per run
- Per-round regret vs. optimal policy
- Inter-agent reward variance

---

## Tech stack

| Component | Choice |
|-----------|--------|
| LLM inference | [Ollama](https://ollama.com) — local, deterministic, no API cost |
| Primary model | `qwen2.5:7b-instruct` |
| Secondary model | `llama3.1:8b-instruct` (ablation) |
| Embedder | `nomic-embed-text` via Ollama |
| Config | YAML + Pydantic |
| Logging | Append-only JSONL per run |
| Analysis | pandas, scipy, matplotlib, seaborn |
| Language | Python 3.11+ |

All experiments are designed to run on a single consumer or research GPU. Deterministic decoding (`temperature=0`) is used for the main experiment; a sampled-decoding control condition is included as an ablation.

---

## Project status

> **In active development.** The codebase does not yet exist — this README documents what is being built.

Implementation follows a phase-gated build order:

| Phase | What gets built | Status |
|-------|----------------|--------|
| 0 | Project scaffold, config system, dependency management | ✅ Completed |
| 1 | Task environments with objective scoring | ✅ Completed |
| 2 | Agent wrapper and Ollama client | ✅ Completed |
| 3 | Three memory policies | ✅ Completed |
| 4 | Logging and reproducibility layer | ✅ Completed |
| 5 | Single-agent baselines | ✅ Completed |
| 6 | Shared broadcast channel | ⬜ Not started |
| 7 | Poisoning condition | ⬜ Not started |
| 8 | Full multi-agent experiment runs | ⬜ Not started |
| 9 | Analysis and paper-ready figures | ⬜ Not started |
| 10 | Writeup | ⬜ Not started |

---

## Why this is interesting

The field has quietly split into "evolve the weights" and "evolve the context", and the context camp is winning on cost-efficiency. But both are hitting the same wall from different mechanistic directions. This project is a small, reproducible, compute-cheap study that runs a direct test of a question the current literature has posed but not answered: when self-evolving agents share memory, do they become more capable, more fragile, or both?

The answer — whatever it is — is a concrete, citable contribution to the gap.

---

## Related work

- **ACE** (Zhang et al., ICLR 2026) — context collapse and brevity bias in memory-based evolution
- **ReasoningBank** (Ouyang et al., Google 2026) — contrastive reasoning memory with test-time scaling
- **Memento** (Zhou et al., UCL 2025) — memory as a learned retrieval policy, not a static store
- **RAGEN** (Wang et al., Northwestern 2025) — Echo Trap in multi-turn RL self-evolution
- **OEP** (Wang et al., SJTU 2026) — locally-correct but non-transferable experience poisoning
- **Zombie Agents** (Yang et al., NUS 2026) — self-reinforcing persistent memory injection
- **EvolveR** (arXiv:2510.16079) — full lifecycle treatment of agent experience
- **Evo-Memory** (arXiv:2511.20857) — benchmark for single-agent memory evolution (the gap this project targets is its multi-agent extension)

---

*August 2026 — Uzair Arif*
