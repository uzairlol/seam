"""Reproducibility helpers: git commit hash, pip freeze, and run ID generation."""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def get_git_commit_hash() -> Optional[str]:
    """Return the current git commit hash (short form), or None if unavailable.

    Returns:
        A 7-character git commit hash string, or None.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        logger.warning("git rev-parse failed (rc=%d): %s", result.returncode, result.stderr.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("Could not determine git commit hash: %s", exc)
    return None


def get_pip_freeze() -> str:
    """Return the output of `pip freeze` as a string.

    Returns:
        Newline-separated installed package list, or an empty string on failure.
    """
    try:
        result = subprocess.run(
            ["pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return result.stdout
        logger.warning("pip freeze failed (rc=%d): %s", result.returncode, result.stderr.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("Could not capture pip freeze: %s", exc)
    return ""


def generate_run_id(experiment_id: str, seed: int) -> str:
    """Generate a unique run identifier.

    Format: ``{experiment_id}_{seed}_{timestamp}`` where *timestamp* is UTC
    in ``%Y%m%dT%H%M%S`` format.

    Args:
        experiment_id: Human-readable experiment identifier.
        seed: Random seed for this run.

    Returns:
        A run ID string suitable for use as a directory or file prefix.
    """
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{experiment_id}_{seed}_{timestamp}"
