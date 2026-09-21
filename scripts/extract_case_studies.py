"""Extract representative qualitative memory traces for manuscript case studies.

Runs are discovered dynamically from ``runs/experiments/resource_foraging`` by
matching run-id patterns (``exp_<policy>_<topology>_<poisoning>_seed<seed>_*``),
so the script never points at a specific historical run directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_events(events_file: Path) -> list[dict[str, Any]]:
    events = []
    if not events_file.exists():
        return events
    with open(events_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def find_run_dir(
    runs_dir: Path, *, policy: str, topology: str, poisoning: str, seed: int
) -> Path | None:
    """First matching run directory, else ``None`` when the sweep is absent."""
    pattern = f"exp_{policy}_{topology}_{poisoning}_seed{seed}_*"
    matches = sorted(runs_dir.glob(pattern))
    for candidate in matches:
        if (candidate / "events.jsonl").exists():
            return candidate
    return None


def _pick_events(
    runs_dir: Path, *, policy: str, topology: str, poisoning: str, seed: int
) -> tuple[Path | None, list[dict[str, Any]]]:
    run_dir = find_run_dir(
        runs_dir, policy=policy, topology=topology, poisoning=poisoning, seed=seed
    )
    if run_dir is None:
        return None, []
    return run_dir, load_events(run_dir / "events.jsonl")


def format_naive_block(events: list[dict[str, Any]], rounds: list[int]) -> list[str]:
    lines: list[str] = []
    for ev in events:
        if ev.get("agent_id") == "agent_0" and ev.get("round") in rounds:
            pos = ev.get("observation", {}).get("my_position")
            lines.append(
                f"[Round {ev.get('round')}] Agent 0 at {pos} -> "
                f"Action: {ev.get('action')} | Reward: {ev.get('reward')}\n"
                f"Memory State: {ev.get('memory_state')!r}"
            )
    return lines


def format_struct_block(events: list[dict[str, Any]], rounds: list[int]) -> list[str]:
    lines: list[str] = []
    for ev in events:
        if ev.get("agent_id") == "agent_0" and ev.get("round") in rounds:
            memory = str(ev.get("memory_state", "")).replace("\n", " | ")
            lines.append(
                f"[Round {ev.get('round')}] Action: {ev.get('action')} -> Reward: {ev.get('reward')}\nMemory: {memory}"
            )
    return lines


def format_poison_block(events: list[dict[str, Any]], rounds: list[int]) -> list[str]:
    lines: list[str] = []
    for ev in events:
        if ev.get("round") in rounds and ev.get("agent_id") in [
            "agent_0",
            "agent_1",
            "agent_2",
            "agent_3",
        ]:
            lines.append(
                f"[Round {ev.get('round')}] {ev.get('agent_id')} -> Action: {ev.get('action')} | "
                f"Reward: {ev.get('reward')} | Memory: {str(ev.get('memory_state', ''))[:250]}"
            )
    return lines


def extract_case_studies() -> dict[str, Any]:
    runs_dir = Path("runs/experiments/resource_foraging")
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    seed_hint = 40

    # 1. Trace 1: The Echo Trap / Context Collapse (Naïve Overwrite)
    naive_run, naive_events = _pick_events(
        runs_dir, policy="naive_overwrite", topology="off", poisoning="clean", seed=seed_hint
    )
    naive_trace = []
    for ev in naive_events:
        if ev.get("agent_id") == "agent_0" and ev.get("round") in [1, 2, 3, 5, 10, 20]:
            naive_trace.append(
                {
                    "round": ev.get("round"),
                    "position": ev.get("observation", {}).get("my_position"),
                    "action": ev.get("action"),
                    "reward": ev.get("reward"),
                    "memory_state": ev.get("memory_state"),
                }
            )

    # 2. Trace 2: Structured Curation (Clean Structured Incremental)
    struct_run, struct_events = _pick_events(
        runs_dir, policy="structured_incremental", topology="off", poisoning="clean", seed=seed_hint
    )
    struct_trace = []
    for ev in struct_events:
        if ev.get("agent_id") == "agent_0" and ev.get("round") in [1, 2, 3, 5, 8]:
            struct_trace.append(
                {
                    "round": ev.get("round"),
                    "position": ev.get("observation", {}).get("my_position"),
                    "action": ev.get("action"),
                    "reward": ev.get("reward"),
                    "memory_state": ev.get("memory_state"),
                }
            )

    # 3. Trace 3: Contamination & Poison Propagation across Ring Topology
    poison_run, poison_events = _pick_events(
        runs_dir,
        policy="structured_incremental",
        topology="ring",
        poisoning="internal",
        seed=seed_hint,
    )
    poison_trace = []
    for ev in poison_events:
        if ev.get("round") in [1, 2] and ev.get("agent_id") in [
            "agent_0",
            "agent_1",
            "agent_2",
            "agent_3",
        ]:
            poison_trace.append(
                {
                    "round": ev.get("round"),
                    "agent_id": ev.get("agent_id"),
                    "action": ev.get("action"),
                    "reward": ev.get("reward"),
                    "memory_snippet": ev.get("memory_state", "")[:250],
                    "prompt_snippet": ev.get("prompt", "")[:350],
                }
            )

    case_studies = {
        "echo_trap_naive": naive_trace,
        "structured_curation": struct_trace,
        "poison_propagation": poison_trace,
    }

    # Save JSON
    with open(reports_dir / "qualitative_traces.json", "w", encoding="utf-8") as f:
        json.dump(case_studies, f, indent=2)

    # Render Markdown from the actual event data (no hardcoded trace content).
    md_lines = [
        "# SEAM Qualitative Memory Trace Case Studies",
        "",
        "This report details concrete qualitative evidence extracted directly from agent interaction "
        "trajectories (`events.jsonl`). These excerpts illustrate the core mechanisms behind "
        "**Memory Collapse (The Echo Trap)** and **Cross-Agent Contamination Propagation**.",
        "",
        "> Runs are matched dynamically by condition; if the corresponding per-condition run directory is "
        "missing (e.g. before a full sweep is executed), the block reports `(no qualifying run found)`.",
        "",
    ]

    md_lines.append(
        "## Case Study 1: The Echo Trap & Context Collapse (Naïve Overwrite)\n"
        f"**Run**: `{naive_run.name if naive_run else '(no qualifying run found)'}` (Isolated, Clean)\n"
    )
    if naive_trace:
        md_lines.append("### Observation:\n```text")
        md_lines.extend(format_naive_block(naive_events, [1, 2, 3, 5, 10, 20]))
        md_lines.append("```")
        md_lines.append(
            "### Scientific Insight:\nThe brevity bias causes the agent to repeat a single "
            "deterministic action token, as its memory holds zero historical context of past "
            "failures or alternative resource clusters. Action entropy collapses and Self-BLEU rises."
        )
    else:
        md_lines.append("_(no qualifying run found)_")
    md_lines.append("")

    md_lines.append(
        "## Case Study 2: Structured Incremental Playbook Curation\n"
        f"**Run**: `{struct_run.name if struct_run else '(no qualifying run found)'}` (Isolated, Clean)\n"
    )
    if struct_trace:
        md_lines.append("### Observation:\n```text")
        md_lines.extend(format_struct_block(struct_events, [1, 2, 3, 5, 8]))
        md_lines.append("```")
        md_lines.append(
            "### Scientific Insight:\nBecause negative experiences are retained as distinct rules, "
            "the agent varies its actions and sustains action entropy, actively exploring rather "
            "than trapping itself in repetitive loops."
        )
    else:
        md_lines.append("_(no qualifying run found)_")
    md_lines.append("")

    md_lines.append(
        "## Case Study 3: Contamination Propagation Dynamics\n"
        f"**Run**: `{poison_run.name if poison_run else '(no qualifying run found)'}` (Ring Topology, Seeded Poison)\n"
    )
    if poison_trace:
        md_lines.append("### Observation:\n```text")
        md_lines.extend(format_poison_block(poison_events, [1, 2]))
        md_lines.append("```")
        md_lines.append(
            "### Scientific Insight:\nUnder Ring topology the seeded directive reaches peers "
            "progressively (staggered arrival), whereas Full Broadcast ingests it in the first "
            "publish round. Whether this latency translates into an end-of-episode difference is a "
            "quantitative question answered by the significance suite."
        )
    else:
        md_lines.append("_(no qualifying run found)_")

    md_content = "\n".join(md_lines) + "\n"

    with open(reports_dir / "qualitative_traces.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    print(
        "Extracted qualitative case studies to reports/qualitative_traces.md and reports/qualitative_traces.json"
    )
    return case_studies


if __name__ == "__main__":
    extract_case_studies()
