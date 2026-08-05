"""IO helpers: loading YAML files and reading/writing JSONL."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Generator

import yaml

logger = logging.getLogger(__name__)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        FileNotFoundError: If *path* does not exist.
        yaml.YAMLError: If the file cannot be parsed.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML file not found: {p}")
    logger.debug("Loading YAML from %s", p)
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)  # type: ignore[no-any-return]


def write_jsonl(records: list[dict[str, Any]], path: str | Path, *, append: bool = True) -> None:
    """Write a list of dicts to a JSONL file.

    Args:
        records: List of JSON-serializable dicts.
        path: Destination file path.
        append: If True, append to an existing file; otherwise overwrite.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with p.open(mode, encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.debug("Wrote %d records to %s (append=%s)", len(records), p, append)


def append_jsonl(record: dict[str, Any], path: str | Path) -> None:
    """Append a single dict to a JSONL file.

    Args:
        record: A JSON-serializable dict.
        path: Destination file path.
    """
    write_jsonl([record], path, append=True)


def read_jsonl(path: str | Path) -> Generator[dict[str, Any], None, None]:
    """Lazily read records from a JSONL file.

    Args:
        path: Path to the JSONL file.

    Yields:
        Dicts parsed from each line.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON on line %d of %s: %s", line_num, p, exc)
