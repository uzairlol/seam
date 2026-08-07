"""RunRehydrator for loading and parsing saved experiment run logs into dataframes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from seam.utils.io import load_yaml, read_jsonl

logger = logging.getLogger(__name__)


class RunRehydrator:
    """Loads metadata, config snapshots, and JSONL step events from a saved run directory.

    Args:
        run_dir: Path to the run directory.
    """

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {self.run_dir}")

        self.metadata_file = self.run_dir / "metadata.json"
        self.config_file = self.run_dir / "config_snapshot.yaml"
        self.events_file = self.run_dir / "events.jsonl"
        self.summary_file = self.run_dir / "summary.json"

    def load_metadata(self) -> dict[str, Any]:
        """Load metadata.json content."""
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file missing in {self.run_dir}")
        import json
        return json.loads(self.metadata_file.read_text(encoding="utf-8"))  # type: ignore[no-any-return]

    def load_config(self) -> dict[str, Any]:
        """Load config_snapshot.yaml content."""
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config snapshot missing in {self.run_dir}")
        return load_yaml(self.config_file)

    def load_events(self) -> list[dict[str, Any]]:
        """Load all step events from events.jsonl."""
        if not self.events_file.exists():
            return []
        return list(read_jsonl(self.events_file))

    def load_summary(self) -> dict[str, Any]:
        """Load summary.json if available."""
        if not self.summary_file.exists():
            return {}
        import json
        return json.loads(self.summary_file.read_text(encoding="utf-8"))  # type: ignore[no-any-return]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all step events into a pandas DataFrame.

        Returns:
            DataFrame with columns for round, agent_id, action, reward, latency_ms, etc.
        """
        events = self.load_events()
        if not events:
            return pd.DataFrame()
        return pd.DataFrame(events)
