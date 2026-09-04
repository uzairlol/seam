# SEAM Documentation

Shared Evolving Agent Memory (SEAM) studies how memory-update policies and peer-memory communication affect task performance, memory collapse, and poisoning propagation in self-evolving LLM agents.

## Start here

1. [Project overview](overview.md) for the research problem, scope, terminology, and repository map.
2. [Architecture](architecture.md) for the components and their data flow.
3. [Experiment lifecycle](experiment-lifecycle.md) for the exact order of operations in a run.
4. [Scenarios](scenarios.md) for the task rules, actions, rewards, and scoring.
5. [Memory policies](memory-policies.md) for the three memory mechanisms.
6. [Sharing and poisoning](sharing-and-poisoning.md) for topologies, routing, injection, and contamination.
7. [Metrics and statistics](metrics-and-statistics.md) for implemented measures and aggregation.
8. [Outputs and reproducibility](outputs-and-reproducibility.md) for commands, artifacts, and rerun checks.
9. [Literature map](literature-map.md) for the connection between the local corpus and SEAM design choices.
10. [Manuscript guide](manuscript-guide.md) for a suggested paper structure and claims discipline.
11. [Audit history](audit-history.md) for known pre-fix result risks and validation status.

## Source of truth

The implementation under `src/seam/` and the executable scripts under `scripts/` are authoritative for current behavior. The root [README](../README.md) contains the research narrative, but some high-level claims and command examples may describe earlier or intended interfaces. When a discrepancy exists, verify against the code and run artifacts.
