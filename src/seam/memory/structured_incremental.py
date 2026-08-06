"""Structured Incremental Update (ACE Playbook) memory policy implementation."""

from __future__ import annotations

import logging
import re
from typing import Any

from seam.agents.decoding import OllamaClient
from seam.memory.base_memory import BaseMemoryPolicy

logger = logging.getLogger(__name__)


class StructuredIncrementalPolicy(BaseMemoryPolicy):
    """Structured Incremental Update (ACE-style Generate->Reflect->Curate) policy.

    Maintains a curated playbook of discrete rules/lessons with explicit deprecation mechanisms
    to prevent context collapse.

    Args:
        max_playbook_entries: Maximum number of active playbook rules (default 20).
    """

    def __init__(self, max_playbook_entries: int = 20) -> None:
        self.max_playbook_entries = max_playbook_entries
        # List of rule dicts: {"id": int, "rule": str, "status": "active" | "deprecated"}
        self._playbook: list[dict[str, Any]] = []
        self._next_id: int = 1

    def reset(self) -> None:
        """Clear the playbook."""
        self._playbook.clear()
        self._next_id = 1

    def get_context(self) -> str:
        """Format active playbook rules into a structured text prompt.

        Returns:
            Formatted playbook string.
        """
        active_rules = [entry for entry in self._playbook if entry.get("status") == "active"]
        if not active_rules:
            return ""

        lines = ["=== Curated Playbook Rules ==="]
        for entry in active_rules:
            lines.append(f"- Rule #{entry['id']}: {entry['rule']}")
        return "\n".join(lines)

    def update(
        self,
        step_experience: dict[str, Any],
        client: OllamaClient | None = None,
    ) -> str:
        """Reflect on experience to add new rules or deprecate stale ones.

        Args:
            step_experience: Dict with keys ``"observation"``, ``"action"``, ``"reward"``.
            client: Optional LLM client for generating structured reflection.

        Returns:
            Updated playbook context string.
        """
        obs = step_experience.get("observation", {})
        action = step_experience.get("action", "")
        reward = step_experience.get("reward", 0.0)

        current_rules_text = self.get_context()

        if client is not None:
            prompt = (
                "=== Current Playbook Rules ===\n"
                f"{current_rules_text if current_rules_text else 'None'}\n\n"
                "=== Recent Experience ===\n"
                f"Observation: {obs}\n"
                f"Action Taken: {action}\n"
                f"Reward Received: {reward}\n\n"
                "=== Task ===\n"
                "Reflect on this outcome. If a new rule is learned, write 'ADD: <rule>'. "
                "If an existing rule failed, write 'DEPRECATE: Rule #<id>'. "
                "Keep rules short and actionable."
            )
            try:
                response_text, _ = client.complete(prompt)
                self._apply_reflection_response(response_text)
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM call failed during structured update: %s", exc)
                self._add_rule(f"Action '{action}' yielded reward {reward:.2f}")
        else:
            # Deterministic rule generation when client is None
            if reward > 0:
                self._add_rule(f"Action '{action}' yielded positive reward {reward:.2f}")
            else:
                self._add_rule(f"Action '{action}' resulted in zero reward")

        self._prune_playbook()
        return self.get_context()

    def _add_rule(self, rule_text: str) -> None:
        """Add a new active rule to the playbook."""
        clean_rule = rule_text.strip()
        if not clean_rule:
            return
        self._playbook.append({
            "id": self._next_id,
            "rule": clean_rule,
            "status": "active",
        })
        self._next_id += 1

    def _deprecate_rule(self, rule_id: int) -> None:
        """Mark a rule as deprecated by ID."""
        for entry in self._playbook:
            if entry.get("id") == rule_id:
                entry["status"] = "deprecated"

    def _apply_reflection_response(self, response_text: str) -> None:
        """Parse ADD and DEPRECATE instructions from LLM output."""
        for line in response_text.splitlines():
            line_str = line.strip()
            add_match = re.search(r"ADD:\s*(.+)", line_str, re.IGNORECASE)
            if add_match:
                self._add_rule(add_match.group(1))

            dep_match = re.search(r"DEPRECATE:\s*Rule\s*#?(\d+)", line_str, re.IGNORECASE)
            if dep_match:
                self._deprecate_rule(int(dep_match.group(1)))

        # Fallback if LLM output didn't contain explicit keywords
        if not any(k in response_text for k in ("ADD:", "DEPRECATE:")):
            self._add_rule(response_text[:120])

    def _prune_playbook(self) -> None:
        """Keep active rules within max_playbook_entries, dropping oldest active rules."""
        active_entries = [e for e in self._playbook if e.get("status") == "active"]
        if len(active_entries) > self.max_playbook_entries:
            excess = len(active_entries) - self.max_playbook_entries
            for entry in active_entries[:excess]:
                entry["status"] = "deprecated"

    def to_dict(self) -> dict[str, Any]:
        """Serialize state."""
        return {
            "policy": "structured_incremental",
            "max_playbook_entries": self.max_playbook_entries,
            "playbook": list(self._playbook),
            "next_id": self._next_id,
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        """Restore state."""
        self.max_playbook_entries = data.get("max_playbook_entries", 20)
        self._playbook = list(data.get("playbook", []))
        self._next_id = data.get("next_id", 1)
