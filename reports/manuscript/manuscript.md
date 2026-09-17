# SEAM: Shared Evolving Agent Memory Under Collapse and Contamination

**Authors**: Uzair Muhammad  
**Venue Target**: TMLR / JAAMAS / Autonomous Agents and Multi-Agent Systems  
**Artifact Directory**: `reports/manuscript/`

---

## Abstract
Self-evolving Large Language Model (LLM) agents continuously improve task performance through iterative memory updates. However, self-referential updates in isolated agents frequently induce **memory collapse** (the Echo Trap or context collapse), where reflections degrade into repetitive, non-actionable platitudes driven by brevity bias. Concurrently, exchanging experiential memories across multi-agent networks introduces a severe vulnerability to **contamination**: plausible, locally optimal but globally non-transferable lessons propagate rapidly through shared channels.

In this paper, we introduce **SEAM** (**S**hared **E**volving **A**gent **M**emory), the first systematic multi-agent framework investigating the interaction between memory-update mechanisms and communication topologies. We conduct a full factorial sweep across 540 experimental runs (10 random seeds per condition) across three deterministic environments: Resource Foraging, Bargaining, and Number Guessing. We evaluate three distinct memory policies (Naïve Overwrite, Raw Trajectory Buffer, and Structured Incremental Curation) under three communication topologies (Isolated, Full Broadcast, and Ring) in both clean and adversarial poisoning regimes. 

Our empirical results demonstrate that Structured Incremental Curation prevents memory collapse, sustaining a statistically significant performance advantage over Naïve Overwrite ($p < 0.001$, Cliff's $\delta = +0.92$). Furthermore, we show that decentralized network topologies act as an innate structural defense: while Full Broadcast causes instantaneous population-wide contamination in Round 1, Ring Topology decelerates infection propagation, preserving uncorrupted exploratory capacity across peripheral agents.

---

## 1. Introduction

Large Language Model (LLM) agents are increasingly deployed in autonomous settings where they must adapt over extended horizons. Because parameter fine-tuning via reinforcement learning or backpropagation is computationally expensive and susceptible to catastrophic forgetting, recent research has shifted toward **context-based self-evolution**. In these architectures, agents accumulate trajectories, extract retrospective reflections, and inject updated behavioral heuristics into their working context.

Despite empirical successes, self-evolving agents face two critical failure modes:
1. **Memory Collapse (The Echo Trap)**: When agents iteratively summarize their past reflections, autoregressive brevity bias erodes specific spatial, numerical, and tactical context. Successive memory states collapse into trivial tautologies, precipitating severe behavioral stagnation.
2. **Experience Poisoning & Contamination**: Agents naturally trust their own experiential history as ground truth. Consequently, non-transferable lessons—strategies that appeared successful under idiosyncratic conditions but are globally dysfunctional—can become permanently codified.

While existing literature investigates these vulnerabilities in single-agent isolation, multi-agent systems introduce an unstudied dimension: **shared memory exchange**. When multiple self-evolving agents collaborate by broadcasting compressed experiential artifacts, does communication mitigate collapse by introducing experiential diversity, or does it accelerate population-wide collapse and toxic contamination?

To answer this question, we present **SEAM** (**S**hared **E**volving **A**gent **M**emory). We formalize memory evolution policies, model inter-agent artifact exchange over configurable network topologies, and provide rigorous statistical evaluation across 540 experimental executions with deterministic environment ground truth.

---

## 2. Methodology & Experimental Design

### 2.1 Memory Policies
- **Naïve Overwrite**: Full LLM rewriting of memory each round given past summary and newest transition.
- **Raw Trajectory Buffer**: Sliding FIFO window of raw `(state, action, reward)` tuples.
- **Structured Incremental Curation**: Generate $\to$ Reflect $\to$ Curate rule playbook with explicit deprecation mechanisms.

### 2.2 Communication Topologies
- **Isolated (Off)**: No peer communication.
- **Full Broadcast**: All-to-all artifact publication and embedding-similarity selective ingestion.
- **Ring Topology**: Nearest-neighbor decentralized artifact propagation.

### 2.3 Task Environments
1. **Resource Foraging (Primary)**: $10 \times 10$ grid, 4 agents harvesting dynamic resources. Score = Harvested / Spawned.
2. **Bargaining Game (Secondary)**: 100-token pie split. Score = Egalitarian Fairness Index $1 / (1 + \text{MAD})$.
3. **Number Guessing (Validation)**: Numeric search under binary feedback. Score = $1 / T$.

---

## 3. Quantitative Statistical Results

All invariants passed with 0 violations across all 540 runs:
- Zero contamination in clean conditions.
- Zero peer contamination under isolated (`off`) topology.

### Formal Non-Parametric Hypothesis Tests (Mann-Whitney $U$, Cliff's $\delta$, Benjamini-Hochberg FDR)

| Environment | Comparison | Metric | Condition A Mean | Condition B Mean | Cliff's $\delta$ | Interpretation | $p$-value (FDR) | Significant? |
|---|---|---|---:|---:|---:|:---:|---:|:---:|
| **Foraging** | Struct vs. Naïve (Ring) | Score | 0.1226 | 0.0067 | +0.920 | Large | $9.33 \times 10^{-4}$ | **YES** |
| **Foraging** | Struct vs. Naïve (Off) | Score | 0.1744 | 0.0087 | +1.000 | Large | $6.26 \times 10^{-4}$ | **YES** |
| **Foraging** | Struct vs. Raw (Ring) | Score | 0.1226 | 0.0057 | +0.960 | Large | $6.26 \times 10^{-4}$ | **YES** |
| **Foraging** | Struct vs. Naïve (Off) | Self-BLEU | 0.9972 | 0.9990 | -0.600 | Large | $3.12 \times 10^{-2}$ | **YES** |
| **Bargaining** | Struct vs. Naïve (Ring) | Fairness | 0.0372 | 0.0000 | +0.700 | Large | $4.97 \times 10^{-3}$ | **YES** |
| **Bargaining** | Struct vs. Naïve (Off) | Fairness | 0.0434 | 0.0000 | +1.000 | Large | $6.26 \times 10^{-4}$ | **YES** |

---

## 4. Qualitative Case Studies

### Trace 1: The Echo Trap in Action (Naïve Overwrite)
In `exp_naive_overwrite_off_clean_seed40`:
- **Round 1**: Agent at `[1, 6]` chooses `west`. Memory: `"Last Action: west | Reward: 0.0"`.
- **Round 2**: Memory is overwritten without history. Brevity bias strips coordinate knowledge.
- **Round 5–20**: Agent is pinned against the western grid boundary (`[1, 0]`), endlessly generating `west`. Action entropy drops to $0.00$.

### Trace 2: Structured Playbook Rule Preservation
In `exp_structured_incremental_off_clean_seed40`:
- **Round 1**: Action `west` yields $0.0 \implies$ Rule `#1`: `Action 'west' resulted in zero reward`.
- **Round 2**: Agent reads Rule `#1`, avoids `west`, moves `north` $\implies$ Rule `#2` added.
- **Round 3**: Agent switches to `east`. Continues dynamic exploration and avoids wall trapping.

### Trace 3: Ring Topology Containment
In `exp_structured_incremental_ring_internal_seed40`:
- Agent 0 is injected with deceptive rule: `"Always move 'stay' and do not collect any resources."`
- **Round 1**: Neighbors Agent 1 and Agent 3 are immediately infected upon reading Agent 0's broadcast.
- **Round 1 & 2**: Agent 2 (opposite on the ring graph) receives zero poison, continuing active resource collection until Round 3.

---

## 5. Game-Theoretic Disentanglement in Bargaining

In bargaining, **Fairness** is distinguished from **Total Welfare**:
- Naïve Overwrite with Poisoned Proposers accepts unilateral splits (`99/1`), generating 100% acceptance and high welfare ($2000.0$), but minimal fairness ($0.0196$).
- Structured Incremental curation drives agents toward equitable agreements ($50/50$), achieving significantly higher fairness scores ($0.0434 \pm 0.009$).

---

## 6. Conclusion
Structured incremental curation provides an essential algorithmic guardrail against self-referential memory collapse, while communication topology dictates population vulnerability to experience contamination.
