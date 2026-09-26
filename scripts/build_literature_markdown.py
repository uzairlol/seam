"""Convert every PDF in literature/ to Markdown in literature/markdown/.

Titles come from literature/papers_metadata.json (produced by
extract_paper_titles.py). Each generated file gets a provenance header and an
H1 title; the duplicated title block at the top of the extracted text is
removed so the heading is the single source of truth.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pymupdf
import pymupdf4llm

ROOT = Path(__file__).resolve().parent.parent
LITERATURE_DIR = ROOT / "literature"
MARKDOWN_DIR = LITERATURE_DIR / "markdown"
METADATA_PATH = LITERATURE_DIR / "papers_metadata.json"
MIN_CHARS = 2_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Reconvert existing Markdown.")
    parser.add_argument(
        "--headers-only",
        action="store_true",
        help=(
            "Rewrite only the provenance header of existing Markdown, reusing the extracted "
            "body. Use after a title or metadata change instead of a full reconversion."
        ),
    )
    return parser.parse_args()


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def strip_leading_title_block(body: str, title: str) -> str:
    """Drop the source PDF's own title block from the top of the extracted text.

    pymupdf4llm renders the PDF's title as a markdown heading, so the text is
    re-decorated before comparison. Leading blank lines are tolerated, and
    multi-line titles are handled by consuming chunks in order.
    """
    target = normalise(title)
    lines = body.split("\n")

    index = 0
    position = 0
    while index < len(lines) and position < len(target):
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            continue
        norm = normalise(stripped)
        if not norm:
            index += 1
            continue
        if not target.startswith(norm, position) and norm != target:
            break
        position += len(norm)
        index += 1

    return "\n".join(lines[index:]).lstrip("\n")


def build_header(title: str, stem: str, abs_url: str, year: str) -> str:
    lines = [f"# {title}", ""]
    if year:
        lines.append(f"**Year:** {year}")
    if abs_url:
        lines.append(f"**arXiv:** [{stem}]({abs_url})")
    lines.append(f"**Source PDF:** `literature/{stem}.pdf`")
    lines.append(
        f"**Converted:** {time.strftime('%Y-%m-%d')} with `pymupdf4llm` "
        f"(PyMuPDF {pymupdf.__version__})"
    )
    lines += ["", "---", ""]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    metadata = {
        record["id"]: record
        for record in json.loads(METADATA_PATH.read_text(encoding="utf-8"))["papers"]
    }
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(LITERATURE_DIR.glob("*.pdf"))
    converted, skipped, failed, refreshed = 0, 0, [], 0

    for index, pdf_path in enumerate(pdf_paths, start=1):
        stem = pdf_path.stem
        target = MARKDOWN_DIR / f"{stem}.md"
        record = metadata.get(stem, {})
        title = record.get("title") or stem

        if args.headers_only:
            if not target.exists():
                print(f"[{index}/{len(pdf_paths)}] {stem}: missing, skipping")
                continue
            current = target.read_text(encoding="utf-8")
            _, _, body = current.partition("\n---\n")
            if not body:
                body = current.split("---", 1)[-1]
            header = build_header(title, stem, record.get("abs_url", ""), record.get("year", ""))
            target.write_text(header + body.lstrip("\n"), encoding="utf-8")
            refreshed += 1
            print(f"[{index}/{len(pdf_paths)}] {stem}: header refreshed")
            continue

        if target.exists() and not args.force and target.stat().st_size > MIN_CHARS:
            skipped += 1
            print(f"[{index}/{len(pdf_paths)}] {stem}: exists, skipping")
            continue

        title = record.get("title") or stem
        title = record.get("title") or stem
        print(f"[{index}/{len(pdf_paths)}] {stem}: converting ...", flush=True)
        try:
            raw = pymupdf4llm.to_markdown(str(pdf_path))
        except Exception as error:  # noqa: BLE001 - report and continue
            failed.append(stem)
            print(f"    FAILED: {type(error).__name__}: {error}", flush=True)
            continue

        body = strip_leading_title_block(raw, title)
        if len(body) < MIN_CHARS:
            failed.append(stem)
            print(f"    FAILED: output too short ({len(body)} chars)", flush=True)
            continue

        header = build_header(title, stem, record.get("abs_url", ""), record.get("year", ""))
        target.write_text(header + body + "\n", encoding="utf-8")
        converted += 1
        print(f"    done: {len(body):,} chars", flush=True)

    print(
        f"\n{converted} converted, {refreshed} headers refreshed, {skipped} skipped, "
        f"{len(failed)} failed" + (f": {failed}" if failed else "")
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
