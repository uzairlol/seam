"""Heuristically recover paper titles from the PDFs in literature/.

Titles come from embedded PDF metadata when usable, otherwise from the first
plausible title lines on page 1. Results are written to
literature/papers_metadata.json for review, and the build step applies any
entries listed in TITLE_OVERRIDES.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pymupdf

LITERATURE_DIR = Path(__file__).resolve().parent.parent / "literature"
OUTPUT_PATH = LITERATURE_DIR / "papers_metadata.json"

# Non-arXiv identifiers whose publication year cannot be read off the ID.
YEAR_OVERRIDES: dict[str, str] = {
    "techrxiv.177203250.05832634_v2": "2026",
}

TITLE_OVERRIDES: dict[str, str] = {
    "2310.11667": ("SOTOPIA: Interactive Evaluation for Social Intelligence in Language Agents"),
    "2407.02209": "Generative Monoculture in Large Language Models",
    "2502.18865": (
        "A Theoretical Perspective: How to Prevent Model Collapse in Self-consuming Training Loops"
    ),
    "2504.20073v2": (
        "RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning"
    ),
    "2505.23352": (
        "Understanding the Information Propagation Effects of Communication Topologies "
        "in LLM-based Multi-Agent Systems"
    ),
    # PDF metadata drops the umlaut and the space after the colon.
    "2505.22954v2": "Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents",
    "2604.17308v1": (
        "SkillFlow: Benchmarking Lifelong Skill Discovery and Evolution for Autonomous Agents"
    ),
}

SKIP_PATTERNS = re.compile(
    r"^(published as|proceedings of|accepted (at|to)|preprint|arxiv:|under review|"
    r"workshop|conference track|anonymous authors|\d{1,2}\s+\w+\s+\d{4}|"
    r"copyright|all rights reserved|abstract$|introduction$)",
    re.IGNORECASE,
)
AUTHOR_HINT = re.compile(r"(@|university|institute|laborator|inc\.|\\\{|email)")
UNDERLINE = re.compile(r"^[\s\-=_*~#]+$")


def looks_like_venue_or_date(line: str) -> bool:
    if SKIP_PATTERNS.match(line.strip()):
        return True
    if re.fullmatch(r"\d{4}", line.strip()):
        return True
    return bool(re.search(r"\b(19|20)\d{2}\b\s*[.,]?\s*$", line.strip()))


def candidate_lines(page_text: str, limit: int = 24) -> list[str]:
    lines = []
    for raw in page_text.split("\n"):
        line = " ".join(raw.split())
        if not line or UNDERLINE.match(line):
            continue
        if looks_like_venue_or_date(line):
            continue
        lines.append(line)
        if len(lines) >= limit:
            break
    return lines


def title_from_page_one(page_text: str) -> str:
    lines = candidate_lines(page_text)
    if not lines:
        return ""

    # A single line with a colon is the strongest signal.
    for line in lines[:8]:
        if ":" in line and len(line) >= 20 and not AUTHOR_HINT.search(line):
            return line

    # Otherwise join consecutive lines until we hit something author-shaped.
    collected: list[str] = []
    for line in lines[:4]:
        if AUTHOR_HINT.search(line):
            break
        if len(line) < 12 and collected:
            break
        collected.append(line)
        joined = " ".join(collected)
        if len(joined) >= 24:
            return joined
    return " ".join(collected) if collected else lines[0]


def tidy(title: str) -> str:
    title = " ".join(title.split())
    title = re.sub(r"\s*\*+\s*", "", title)
    title = re.sub(r"\s*([.,:])$", "", title)
    return title.strip()


def year_from_page_one(page_text: str) -> str:
    """Find a written-out date such as '27 February 2026' on the title page."""
    match = re.search(
        r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+(\d{4})\b",
        page_text,
    )
    if not match:
        return ""
    return match.group(3)


def year_for(stem: str, page_one: str) -> str:
    if stem in YEAR_OVERRIDES:
        return YEAR_OVERRIDES[stem]
    match = re.match(r"^(\d{2})(\d{2})\.\d{4,5}", stem)
    if match:
        return f"20{match.group(1)}"
    return year_from_page_one(page_one)


def main() -> int:
    records = []
    for pdf_path in sorted(LITERATURE_DIR.glob("*.pdf")):
        stem = pdf_path.stem
        doc = pymupdf.open(pdf_path)
        meta_title = tidy((doc.metadata or {}).get("title", "") or "")
        page_one = doc[0].get_text()
        doc.close()

        title = TITLE_OVERRIDES.get(stem) or ""
        source = "override"
        if not title and 12 <= len(meta_title) <= 300:
            title, source = meta_title, "pdf-metadata"
        if not title:
            title, source = tidy(title_from_page_one(page_one)), "page-one"

        records.append(
            {
                "id": stem,
                "title": title,
                "title_source": source,
                "year": year_for(stem, page_one),
                "abs_url": f"https://arxiv.org/abs/{stem.split('v')[0]}"
                if re.match(r"^\d{4}\.\d{4,5}", stem)
                else "",
            }
        )
        print(f"{stem:<38} [{source:<13}] {title[:88]}")

    OUTPUT_PATH.write_text(
        json.dumps({"count": len(records), "papers": records}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nWrote {OUTPUT_PATH} with {len(records)} records.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
