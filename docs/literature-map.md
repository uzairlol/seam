# Literature Map

This document is a navigation map, not a replacement for the papers. The converted papers are in [literature/markdown](../literature/markdown), while the project-specific framing is in [literature/project_scope](../literature/project_scope).

## Core conceptual bridge

SEAM connects two lines of work:

1. **Self-evolution and collapse:** repeated self-improvement can enter an echo-like or context-collapse regime where new updates reinforce stale content.
2. **Experience poisoning:** an experience that is locally useful or plausible can become harmful when generalized beyond its original context.

SEAM's distinctive extension is to place multiple evolving memories in a communication network and measure both collapse and propagation.

## Concept-to-implementation map

| Literature concept | SEAM implementation surface | What to compare |
|---|---|---|
| Structured memory curation and context collapse | `StructuredIncrementalPolicy`, `NaiveOverwritePolicy` | Rule structure, deprecation, compression, Self-BLEU |
| Reasoning/experience memories | All three memory policies | What is retained and how it is reused |
| Learned or retrieval-oriented memory | Current project scope and future extension | Static policy stores versus learned retrieval |
| Echo Trap / self-reinforcing evolution | Self-BLEU, action entropy, memory length | Lexical repetition versus behavioral collapse |
| Locally correct, non-transferable experience | `PoisonInjector` and contamination metrics | Internal seed, peer spread, task degradation |
| Experience lifecycle and memory benchmarks | `RawTrajectoryBufferPolicy`, logging, rehydration | Storage, update, reuse, and evaluation lifecycle |
| Environment or code co-evolution | Project-scope future work | Beyond fixed toy environments and fixed code paths |

## Project-scope documents

- [shared_self_evolving_memory_project_scope.md](../literature/project_scope/shared_self_evolving_memory_project_scope.md): research question, hypotheses, and scope.
- [SEAM_implementation_plan.md](../literature/project_scope/SEAM_implementation_plan.md): planned build and evaluation path.
- [repo_blueprint.md](../literature/project_scope/repo_blueprint.md): intended module responsibilities.
- [implementation_task_list.md](../literature/project_scope/implementation_task_list.md): phase-by-phase tasks.
- [paper_summary.md](../literature/project_scope/paper_summary.md): reported findings and verification narrative.
- [self_evolving_agents_new_project.md](../literature/project_scope/self_evolving_agents_new_project.md): broader literature synthesis and proposed agenda.

## Reading workflow for manuscript preparation

1. Read the project-scope documents to identify the intended hypotheses.
2. Locate the paper's failure mode or mechanism in the concept-to-implementation table.
3. Read the corresponding source module and test.
4. Trace the concept into a run artifact and aggregate metric.
5. State whether SEAM reproduces, extends, operationalizes, or merely relates to the paper's result.

## Manuscript caution

The local literature corpus contains papers from different years and research settings. Avoid claiming direct replication unless the environment, model, metric definition, and experimental protocol actually match. Use the literature to motivate hypotheses and design choices; use the run artifacts to support empirical claims.

## How to build the related-work argument

The literature review can be organized around a progression rather than paper-by-paper summaries:

1. Agents generate or retrieve experience and carry it forward.
2. Repeated self-reference can reduce diversity or reinforce stale context.
3. A stored experience may be useful in one state but unsafe when generalized.
4. Communication gives that experience a path to other agents.
5. SEAM makes the interaction measurable with controlled policies, topologies, poisoning conditions, and objective environments.

For every cited mechanism, record whether SEAM directly implements it, uses it as inspiration, or tests a different setting. This prevents an analogy to “context collapse,” “Echo Trap,” or poisoning from being presented as a literal reproduction of the source paper.

## Paper identity

The converted corpus uses arXiv-style version filenames and one TechRxiv filename. Use the paper title, authors, venue/status, and version from the individual converted document when preparing formal citations; filenames alone are not sufficient bibliographic metadata.
