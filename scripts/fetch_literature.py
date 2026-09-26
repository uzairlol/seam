"""Download arXiv PDFs into literature/ with polite rate limiting and validation."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

LITERATURE_DIR = Path(__file__).resolve().parent.parent / "literature"
USER_AGENT = "SEAM-Literature/1.0 (research literature collection; contact: local user)"
DELAY_SECONDS = 3.0
MAX_ATTEMPTS = 4
MIN_PDF_BYTES = 20_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ids",
        nargs="+",
        required=True,
        help="arXiv IDs to fetch, e.g. 2506.07398 2605.22721",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a valid PDF already exists.",
    )
    return parser.parse_args()


def is_valid_pdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < MIN_PDF_BYTES:
        return False
    with path.open("rb") as handle:
        return handle.read(5) == b"%PDF-"


def download(arxiv_id: str) -> tuple[Path, str]:
    target = LITERATURE_DIR / f"{arxiv_id}.pdf"
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    last_error = "unknown error"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read()
            if len(payload) < MIN_PDF_BYTES:
                last_error = f"suspiciously small payload ({len(payload)} bytes)"
                time.sleep(DELAY_SECONDS * attempt)
                continue
            if payload[:5] != b"%PDF-":
                last_error = f"not a PDF, first bytes were {payload[:16]!r}"
                time.sleep(DELAY_SECONDS * attempt)
                continue
            target.write_bytes(payload)
            return target, "downloaded"
        except urllib.error.HTTPError as error:
            last_error = f"HTTP {error.code}"
            if error.code in (404, 403):
                break
            time.sleep(DELAY_SECONDS * attempt)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            last_error = f"{type(error).__name__}: {error}"
            time.sleep(DELAY_SECONDS * attempt)

    return target, f"FAILED ({last_error})"


def main() -> int:
    args = parse_args()
    LITERATURE_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, str] = {}
    for index, arxiv_id in enumerate(args.ids, start=1):
        target = LITERATURE_DIR / f"{arxiv_id}.pdf"
        if not args.force and is_valid_pdf(target):
            results[arxiv_id] = "already present"
            print(f"[{index}/{len(args.ids)}] {arxiv_id}: already present")
            continue

        print(f"[{index}/{len(args.ids)}] {arxiv_id}: fetching ...", flush=True)
        _, status = download(arxiv_id)
        results[arxiv_id] = status
        print(f"[{index}/{len(args.ids)}] {arxiv_id}: {status}", flush=True)
        if status == "downloaded":
            time.sleep(DELAY_SECONDS)

    summary = {"requested": len(args.ids), "results": results}
    print("\n" + json.dumps(summary, indent=2))
    failures = [k for k, v in results.items() if v.startswith("FAILED")]
    print(f"\n{len(args.ids) - len(failures)} ok, {len(failures)} failed: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
