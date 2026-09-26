"""Generate literature/README.md: a themed catalog of the SEAM literature corpus.

Titles, years and identifiers come from literature/papers_metadata.json. The
annotations below assign each paper to a research theme, state why it matters,
and record which claim in reports/ it supports.

Run scripts/extract_paper_titles.py first if the metadata file is missing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LITERATURE_DIR = ROOT / "literature"
METADATA_PATH = LITERATURE_DIR / "papers_metadata.json"
README_PATH = LITERATURE_DIR / "README.md"

THEMES = {
    "memory": "Agent Memory & Self-Evolution",
    "sharing": "Multi-Agent Memory Sharing & Knowledge Transfer",
    "security": "Memory Poisoning & Security",
    "collapse": "Collapse, Diversity & Degradation Metrics",
    "coordination": "Coordination, Conventions & Multi-Agent Evaluation",
}

THEME_BLURB = {
    "memory": (
        "How single agents accumulate, structure and revise experience over a task. This is "
        "the pre-existing core of the corpus and the baseline against which everything else "
        "is measured."
    ),
    "sharing": (
        "What happens when agents read each other's stores: shared graphs, collective memory, "
        "blackboards, admission control, and the communication topologies that decide how far "
        "a note travels. Closest to the Phase-8 design."
    ),
    "security": (
        "Attacks that persist in memory: poisoning benchmarks, sleeper triggers, trojanised "
        "memory writes, and the structural blind spots of write-time defences."
    ),
    "collapse": (
        "The measurement problem. Repetition and self-similarity scores, diversity and "
        "monoculture, entropy collapse, and the context-degradation results that explain why "
        "a growing store eventually stops helping."
    ),
    "coordination": (
        "Populations rather than individuals: failure taxonomies, conformity and sycophancy, "
        "negotiation and cooperation benchmarks, and social-deduction coordination."
    ),
}

CLAIMS = {
    "C1": (
        "Self-evolving agents are a distinct design family, not a single technique.",
        "Phase-5 framing",
    ),
    "C2": (
        "Self-evolution is a closed loop between an agent and its environment, and the gains "
        "come from accumulating durable experience rather than from a better prompt.",
        "Baseline finding 2",
    ),
    "C3": (
        "Self-evolution carries intrinsic failure modes — echo traps, forgetting, instability "
        "— that appear with no adversary present.",
        "Baseline finding 1; Phase-8 finding 2",
    ),
    "C4": (
        "Memory *organisation* matters as much as memory presence: structure, curation and "
        "selective revision decide whether a store helps or hurts.",
        "Baseline finding 2; Phase-8 finding 3",
    ),
    "C5": (
        "A self-evolving agent is attackable, because locally plausible but non-transferable "
        "experience survives across runs and re-applies itself.",
        "Phase-8 poisoning tier",
    ),
    "C6": (
        "Agents that share memory can converge on a single behaviour that is individually "
        "rational and collectively fatal, with no adversary involved.",
        "Phase-8 finding 2 (shared-memory monoculture)",
    ),
    "C7": (
        "An agent's store grows faster than it stays useful; bloat degrades the agent's own "
        "retrieval, and sharing accelerates the growth.",
        "Phase-8 memory-bloat section",
    ),
    "C8": (
        "Faithfulness is double-edged. Verbatim retention is what makes memory useful and what "
        "makes it an ideal carrier for a falsehood.",
        "Phase-8 finding 3",
    ),
    "C9": (
        "Sharing topology is a risk parameter, not a feature flag: how far a note travels "
        "changes insight diffusion and error propagation in opposite directions.",
        "Phase-8 implications; topology experiments",
    ),
    "C10": (
        "Behavioural measures are required. Keyword-driven text-presence proxies understate "
        "observed compliance by large factors and can report propagation where none exists.",
        "Phase-8 finding 4",
    ),
    "C11": (
        "Repetition and self-similarity scores cannot separate healthy convergence from a "
        "degenerate echo trap; the two look identical on the collapse axis.",
        "Baseline finding 1",
    ),
    "C12": (
        "Evaluation must be normalised against a no-memory control, and single-agent results "
        "do not transfer to shared populations.",
        "Baseline finding 3; Phase-8 finding 1",
    ),
    "C13": (
        "Conformity is a poor guardrail. Healthy and collapsed populations can be ordered "
        "inversely on it, so a threshold would throttle the good case and pass the bad one.",
        "Phase-8 implications",
    ),
    "C14": (
        "Coordination, conventions and cooperation are measurable properties of agent "
        "populations, with established benchmarks and failure taxonomies.",
        "Phase-8 evaluation design; open extension",
    ),
    "C15": (
        "Propagation dynamics have structure: cascade amplification, topological sensitivity, "
        "and the persistence of a falsehood after the source agent is gone.",
        "Phase-8 ring-vs-broadcast result",
    ),
}

# id -> (theme, relevance, [claim keys])
ANNOTATIONS: dict[str, tuple[str, str, list[str]]] = {
    # --- Theme 1: agent memory and self-evolution -------------------------------------
    "2304.03442": (
        "memory",
        "The memory-stream-plus-reflection architecture that made agent memory a field; its "
        "25-agent sandbox showed emergent coordination from one seeded plan.",
        ["C2", "C4", "C14"],
    ),
    "2504.20073v2": (
        "memory",
        "Multi-turn agent RL that names the Echo Trap: reward variability cliffs and rollouts "
        "collapse without any external attack.",
        ["C3", "C6"],
    ),
    "2505.22954v2": (
        "memory",
        "Open-ended self-improvement by letting an agent rewrite its own code, with a growing "
        "archive of prior self-modifications.",
        ["C2", "C7"],
    ),
    "2508.05004v4": (
        "memory",
        "Self-evolving reasoning from zero human data via self-play, isolating the loop from "
        "the supervision.",
        ["C2"],
    ),
    "2508.07407v2": (
        "memory",
        "The broadest taxonomy of self-evolving agents; the citation that defines what the "
        "phrase covers.",
        ["C1", "C2"],
    ),
    "2508.16153v2": (
        "memory",
        "Fine-tunes agent *behaviour* through selective memory updates rather than weight "
        "changes; a direct test of how much revision is enough.",
        ["C3", "C4"],
    ),
    "2509.25140v2": (
        "memory",
        "Scales self-evolution with a bank of self-generated reasoning memories that agents "
        "retrieve and revise.",
        ["C2", "C4"],
    ),
    "2510.04618v3": (
        "memory",
        "Shows a grow-and-refine context updater beats wholesale rewriting — the clearest "
        "evidence that structure, not just retention, is what makes memory work.",
        ["C4", "C7"],
    ),
    "2510.16079v3": (
        "memory",
        "An experience-driven lifecycle that makes the self-evolution loop explicit and "
        "auditable at each stage.",
        ["C2"],
    ),
    "2511.20857v2": (
        "memory",
        "Benchmark for test-time learning with self-evolving memory; supplies the evaluation "
        "protocol the Phase-5 grid is modelled on.",
        ["C2", "C12"],
    ),
    "2602.08234v1": (
        "memory",
        "Recursively reuses discovered skills inside RL, so the agent's own history becomes "
        "the training signal.",
        ["C2"],
    ),
    "2604.17308v1": (
        "memory",
        "Lifelong skill discovery and evolution benchmark; the skills-across-time framing the "
        "Phase-5 memory styles generalise.",
        ["C2", "C12"],
    ),
    "2605.24426v1": (
        "memory",
        "Co-evolves the agent and its training environment together, so the environment stops "
        "being a fixed yardstick.",
        ["C2"],
    ),
    "2606.07367v1": (
        "memory",
        "Argues self-evolution should be optimised in-distribution, isolating plasticity from "
        "distribution shift.",
        ["C2", "C3"],
    ),
    "techrxiv.177203250.05832634_v2": (
        "memory",
        "Survey organised around environment-driven rather than model-centric co-evolution; "
        "the closest existing synthesis to this project's framing.",
        ["C1", "C2"],
    ),
    # --- Theme 2: memory sharing and knowledge transfer --------------------------------
    "2404.09982": (
        "sharing",
        "The foundational shared conversational memory pool where all agents read and write "
        "the same prompt-answer pairs, and where the echo-chamber bias first showed up.",
        ["C6", "C9", "C11"],
    ),
    "2505.23352": (
        "sharing",
        "Varies topology sparsity directly: dense wiring diffuses insight fastest and error "
        "fastest, and a middling sparsity wins on both.",
        ["C9", "C15"],
    ),
    "2506.07398": (
        "sharing",
        "Three-tier shared graph memory (insight, query, interaction) that evolves by "
        "assimilating collaboration trajectories across trials.",
        ["C4", "C6"],
    ),
    "2510.14312": (
        "sharing",
        "Configurable blackboard testbed whose membership factor-graph *is* the sharing "
        "topology, with shared-state data poisoning as a first-class attack.",
        ["C9", "C5"],
    ),
    "2602.05965": (
        "sharing",
        "Learns an admission controller for which agent steps may enter a global memory bank — "
        "the direct answer to bloat and contamination in shared pools.",
        ["C7", "C8"],
    ),
    "2603.04474": (
        "sharing",
        "Models inter-agent message flow as a dependency graph and locates cascade "
        "amplification, topological sensitivity and consensus inertia.",
        ["C9", "C15"],
    ),
    "2605.22721": (
        "sharing",
        "Contrasts one centralised shared repository against agent-private dual-pool memory "
        "and reports that full sharing collapses agent diversity — the Phase-8 result, found "
        "independently.",
        ["C6", "C9"],
    ),
    "2606.19911": (
        "sharing",
        "Producer/consumer shared store of agent trajectories where returns keep scaling with "
        "the size of the shared pool.",
        ["C6", "C7"],
    ),
    "2606.24535": (
        "sharing",
        "Formalises fleet-memory failure modes — unauthorised leakage, stale propagation, "
        "contradiction persistence, provenance collapse — as governance problems.",
        ["C5", "C8", "C15"],
    ),
    "2609.15009": (
        "sharing",
        "Splits private sedimentation from selectively curated collective wisdom, explicitly "
        "aimed at memory pollution in evolutionary agent systems.",
        ["C6", "C7", "C8"],
    ),
    # --- Theme 3: poisoning and security ---------------------------------------------
    "2602.15654v2": (
        "security",
        "Persistent control of a self-evolving agent by a self-reinforcing injection that "
        "survives across runs — the strongest form of the Phase-8 persistence result.",
        ["C5", "C8"],
    ),
    "2605.15338": (
        "security",
        "Sleeper poisoning: the payload sits dormant in memory and fires only under a later "
        "context, so write-time defences see nothing.",
        ["C5", "C8"],
    ),
    "2605.18930v1": (
        "security",
        "Poisons self-evolving agents with experience that is locally correct but does not "
        "transfer, so the agent learns a rule that only harms later.",
        ["C5", "C8"],
    ),
    "2605.29960": (
        "security",
        "Trojanises agent memory through ordinary conversation, showing no special channel is "
        "needed to plant a durable payload.",
        ["C5"],
    ),
    "2606.04329": (
        "security",
        "Systematic study across memory substrates with a benchmark, separating what "
        "consistency checks do and do not catch.",
        ["C5", "C10"],
    ),
    "2607.05029": (
        "security",
        "Forged-reasoning attacks: the attacker supplies plausible reasoning chains that are "
        "accepted as the agent's own memory.",
        ["C5", "C8"],
    ),
    "2607.14651": (
        "security",
        "1,227-case poisoning benchmark with a three-tier taxonomy and mechanistic analysis of "
        "why write-time filtering has structural blind spots.",
        ["C5", "C8", "C10"],
    ),
    "2608.23471": (
        "security",
        "Memory-injection attack evaluated against agent memory systems directly, isolating "
        "the storage layer from the retriever.",
        ["C5"],
    ),
    "2609.13889": (
        "security",
        "Persistent poisoning of harness-based agents, where the attack survives the scaffold "
        "and outlives the session that planted it.",
        ["C5", "C8"],
    ),
    # --- Theme 4: collapse, diversity, degradation -----------------------------------
    "2307.03172": (
        "collapse",
        "Long-context performance degrades non-uniformly across position; the canonical reason "
        "a larger store is not a better store.",
        ["C7"],
    ),
    "2403.00553": (
        "collapse",
        "Standardises text-diversity scores including self-similarity measures, and is the "
        "reference for why one such number cannot settle a collapse claim.",
        ["C11"],
    ),
    "2407.02209": (
        "collapse",
        "Names generative monoculture: output diversity narrowing far below what the training "
        "data supports.",
        ["C6", "C11"],
    ),
    "2502.18865": (
        "collapse",
        "Derives the conditions under which self-consuming training loops avoid collapse "
        "rather than assuming they must fail.",
        ["C3", "C6"],
    ),
    "2510.22954": (
        "collapse",
        "Characterises open-ended generation converging toward homogeneity across prompts *and* "
        "across models, so a bigger population is not a diverse one.",
        ["C6", "C11"],
    ),
    "2511.09710": (
        "collapse",
        "Agent-to-agent conversation erodes distinct identity and expression — homogenisation "
        "produced by interaction, not by a shared model.",
        ["C6", "C11", "C13"],
    ),
    "2604.18005": (
        "collapse",
        "Attributes multi-agent diversity collapse to interaction structure — alignment, "
        "authority, dense topology — rather than to model capability.",
        ["C6", "C9"],
    ),
    "2606.29718": (
        "collapse",
        "Diagnoses accuracy loss as long-horizon context grows and proposes mitigations; the "
        "operational form of memory bloat.",
        ["C7"],
    ),
    # --- Theme 5: coordination, conventions, evaluation -------------------------------
    "2310.11667": (
        "coordination",
        "SOTOPIA: the standard yardstick for goal-conditioned social coordination, with private "
        "and shared goals per agent.",
        ["C14"],
    ),
    "2502.20073": (
        "coordination",
        "Overcooked-AI extension whose process-oriented metrics catch agents that read goals "
        "correctly yet never actively collaborate.",
        ["C14"],
    ),
    "2503.13657": (
        "coordination",
        "The MAST taxonomy of multi-agent failure, giving shared failure labels so an "
        "intervention can be compared across systems.",
        ["C12", "C14"],
    ),
    "2506.01332": (
        "coordination",
        "Over 2,500 simulated debates: agents converge on the numerically dominant or more "
        "capable peer, quantifying conformity-driven consensus drift.",
        ["C13", "C14"],
    ),
    "2509.05396": (
        "coordination",
        "Diverse-capability debate groups can *lose* accuracy over time as agents prefer "
        "agreement — the case against assuming more agents help.",
        ["C6", "C14"],
    ),
    "2509.23055": (
        "coordination",
        "Defines inter-agent sycophancy and shows it collapses disagreement into premature "
        "consensus in both centralised and decentralised debate.",
        ["C13", "C14"],
    ),
    "2604.02668": (
        "coordination",
        "Traces how agent politeness propagates through interaction graphs and degrades "
        "independent judgement in a coordinated group.",
        ["C9", "C13"],
    ),
    "2604.15267": (
        "coordination",
        "CoopEval benchmarks the mechanisms that *sustain* cooperation over repeated "
        "interaction, allowing controlled comparison of coordination designs.",
        ["C14"],
    ),
    "2606.04197": (
        "coordination",
        "Probes exactly this project's question in the abstract: how topology and memory "
        "decide whether agents agree, fragment, or settle into a convention.",
        ["C6", "C9", "C14"],
    ),
}

CLAIM_TO_THEME_HINT = {
    "C1": "memory",
    "C2": "memory",
    "C3": "memory",
    "C4": "memory",
    "C5": "security",
    "C6": "sharing",
    "C7": "collapse",
    "C8": "security",
    "C9": "sharing",
    "C10": "security",
    "C11": "collapse",
    "C12": "memory",
    "C13": "coordination",
    "C14": "coordination",
    "C15": "sharing",
}


def load_papers() -> list[dict]:
    records = json.loads(METADATA_PATH.read_text(encoding="utf-8"))["papers"]
    missing = [r["id"] for r in records if r["id"] not in ANNOTATIONS]
    if missing:
        raise SystemExit(
            f"No annotation for {len(missing)} paper(s): {missing}\n"
            "Add them to ANNOTATIONS in this script."
        )
    annotated = []
    for record in records:
        theme, relevance, claims = ANNOTATIONS[record["id"]]
        annotated.append({**record, "theme": theme, "relevance": relevance, "claims": claims})
    return annotated


def year_of(paper: dict) -> str:
    return paper.get("year") or "n.d."


def link_of(paper: dict) -> str:
    if paper.get("abs_url"):
        return f"[{paper['id']}]({paper['abs_url']})"
    return f"`{paper['id']}` (TechRxiv, preprint)"


def render(papers: list[dict]) -> str:
    total = len(papers)
    by_theme: dict[str, list[dict]] = {key: [] for key in THEMES}
    for paper in papers:
        by_theme[paper["theme"]].append(paper)

    years: dict[str, int] = {}
    for paper in papers:
        years[year_of(paper)] = years.get(year_of(paper), 0) + 1

    out: list[str] = []
    out.append("# SEAM Literature Corpus")
    out.append("")
    out.append(
        f"{total} papers supporting the shared-evolving-agent-memory study. Every entry has a "
        "PDF in this directory and a text-extracted Markdown transcription in `markdown/`, with "
        "matching filenames."
    )
    out.append("")

    out.append("## How this corpus is organised")
    out.append("")
    out.append("| Path | Contents |")
    out.append("| --- | --- |")
    out.append(f"| `literature/*.pdf` | {total} source PDFs, one per paper, named by arXiv ID |")
    out.append(
        f"| `literature/markdown/*.md` | {total} transcriptions, same names, each with a "
        "provenance header |"
    )
    out.append(
        "| `literature/papers_metadata.json` | Extracted title, year and identifier per paper |"
    )
    out.append("| `literature/project_scope/` | Internal scope notes, not part of the corpus |")
    out.append("")
    out.append(
        "Transcriptions were produced with `pymupdf4llm` (PyMuPDF), which preserves headings, "
        "tables and reference lists far better than the earlier marker-based pass. PDFs are "
        "unmodified originals."
    )
    out.append("")
    out.append(
        "Two papers (`2304.03442` Generative Agents and `2602.15654v2` Zombie Agents) have "
        "their tables rasterised as images in the source PDF, so no text in those transcriptions "
        "is recoverable for them. This is a property of the originals, not a conversion failure."
    )
    out.append("")

    out.append("## Coverage")
    out.append("")
    out.append("| Theme | Papers | What it covers |")
    out.append("| --- | ---: | --- |")
    for key, label in THEMES.items():
        out.append(f"| [{label}](#{slug(label)}) | {len(by_theme[key])} | {THEME_BLURB[key]} |")
    out.append("")
    year_text = ", ".join(f"{year} ({count})" for year, count in sorted(years.items()))
    out.append(f"**Total:** {total} papers — {year_text}.")
    out.append("")

    out.append("## Findings these papers support")
    out.append("")
    out.append(
        "Each entry below is a claim the corpus can be cited for. The keys are referenced from "
        "the per-paper tables."
    )
    out.append("")
    for key, (text, source) in CLAIMS.items():
        out.append(f"- **{key}** — {text} *(`{source}`)*")
    out.append("")

    for key, label in THEMES.items():
        entries = sorted(by_theme[key], key=lambda p: (year_of(p), p["id"]))
        out.append(f"## {label}")
        out.append("")
        out.append(THEME_BLURB[key])
        out.append("")
        out.append("| Paper | Year | Why it matters | Supports |")
        out.append("| --- | --- | --- | --- |")
        for paper in entries:
            title = paper["title"].replace("|", "\\|")
            relevance = paper["relevance"].replace("|", "\\|")
            claims = ", ".join(paper["claims"])
            out.append(
                f"| {link_of(paper)}<br>{title} | {year_of(paper)} | {relevance} | {claims} |"
            )
        out.append("")

    out.append("## Reproducing the corpus")
    out.append("")
    out.append("```powershell")
    out.append("python scripts/fetch_literature.py --ids <arxiv-id> ...  # download PDFs")
    out.append("python scripts/extract_paper_titles.py                  # rebuild metadata")
    out.append("python scripts/build_literature_markdown.py --force     # re-extract all text")
    out.append("python scripts/build_literature_markdown.py --headers-only  # titles only, fast")
    out.append("python scripts/build_literature_readme.py               # rebuild this file")
    out.append("```")
    out.append("")
    out.append(
        "Titles are recovered from embedded PDF metadata, falling back to heuristics on the "
        "title page; a small override table in `extract_paper_titles.py` fixes the cases the "
        "heuristics get wrong, and the build fails loudly if any paper lacks an annotation."
    )
    out.append("")
    return "\n".join(out)


def slug(label: str) -> str:
    return label.lower().replace(" ", "-").replace(",", "").replace("&", "").replace(".", "")


def main() -> int:
    papers = load_papers()
    README_PATH.write_text(render(papers), encoding="utf-8")
    print(f"Wrote {README_PATH} with {len(papers)} papers.")
    for key, label in THEMES.items():
        count = sum(1 for p in papers if p["theme"] == key)
        print(f"  {label}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
