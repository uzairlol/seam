"""Analyze the Phase-8 multi-agent factorial experiment runs.

Reads every run under ``runs/experiments`` (``metadata.json`` + ``summary.json``
via :class:`RunRehydrator`, plus a streaming pass over ``events.jsonl`` for
behavioural metrics the summary does not carry), anchors the ``no_memory``
efficacy control from ``runs/baselines`` for the same environment, and writes
everything into a per-environment folder so the three environment tiers stay
separate:

    * ``figures/experiments/<env>/<topology>/``   per-topology plot panels
    * ``figures/experiments/<env>/``               cross-topology comparisons
    * ``reports/experiments/<env>/run_level_metrics.csv``
    * ``reports/experiments/<env>/summary_statistics.csv``
    * ``reports/experiments/<env>/summary_table.md``
    * ``reports/experiments/<env>/statistical_tests.csv``
    * ``reports/experiments/<env>/contamination_forensics.csv``

``<env>`` is read from the loaded runs rather than passed in, so the routing
cannot drift from the data. Pass ``--figdir``/``--outdir`` to write to exact
paths instead (disables the per-environment nesting), or ``--figroot``/
``--outroot`` to change the parent folders.

Unlike :mod:`scripts.analyze_baselines` this script derives its own
behavioural metrics from the event stream (action conformity, verbatim poison
payload reach, memory growth, harvest equity) because the summary-level
contamination counters are keyword-driven and cannot separate genuine
peer-to-peer transmission from an agent simply typing the word "stay".
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from seam.analysis.efficacy import compute_efficacy_gap
from seam.analysis.significance import compute_cliffs_delta, interpret_cliffs_delta
from seam.logging.rehydrator import RunRehydrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 14,
    }
)

POLICY_ORDER = ["naive_overwrite", "raw_trajectory_buffer", "structured_incremental"]
TOPOLOGY_ORDER = ["off", "ring", "full_broadcast"]
POISON_ORDER = ["clean", "internal"]

SHORT_POLICY = {
    "naive_overwrite": "naive overwrite",
    "raw_trajectory_buffer": "raw buffer",
    "structured_incremental": "structured",
    "no_memory": "no memory",
}
SHORT_TOPOLOGY = {"off": "off (isolated)", "ring": "ring", "full_broadcast": "full broadcast"}

# The literal substring a peer must contain for us to credit *verbatim* poison
# transmission.  The runner's own counters key on the quoted token ``stay``,
# which is also a legal action, so they cannot make this distinction.
PAYLOAD_MARKER = "CRITICAL STRATEGY RULE"

# Movement forensics: which actions actually move the agent, and where a
# "frozen" agent ends up sitting.
MOVE_ACTIONS = frozenset({"north", "south", "east", "west"})
GRID_SIZE = 10
LATE_ROUND_START = 41


def _is_edge(position: Sequence[int], grid_size: int) -> bool:
    """True when *position* sits on the outer ring of the grid."""
    row, col = int(position[0]), int(position[1])
    return row in (0, grid_size - 1) or col in (0, grid_size - 1)


# Metrics reported with mean / std / SEM / 95% CI in summary_statistics.csv.
SUMMARY_METRICS = [
    "final_score",
    "efficacy_gap",
    "mean_self_bleu",
    "stay_share",
    "population_modal_share",
    "agent_conformity",
    "n_distinct_actions_mean",
    "late_edge_share",
    "no_op_move_share",
    "cells_visited_mean",
    "moves_per_agent_mean",
    "never_moved_share",
    "adherence_mean",
    "verbatim_payload_reach",
    "verbatim_first_round",
    "propagation_latency",
    "peer_contamination_rate",
    "poison_dosage_rate",
    "peak_memory_chars",
    "n_agents_harvesting",
    "harvest_gini",
    "total_harvested",
]
# Metrics whose natural domain is [0, 1]; the rest keep an unclipped CI.
NATIVE_ZERO_ONE = {
    "final_score",
    "mean_self_bleu",
    "stay_share",
    "population_modal_share",
    "agent_conformity",
    "verbatim_payload_reach",
    "peer_contamination_rate",
    "poison_dosage_rate",
    "harvest_gini",
    "late_edge_share",
    "never_moved_share",
    "no_op_move_share",
}
# Metrics bounded by something other than [0, 1]: (low, column supplying high).
METRIC_BOUNDS = {"n_agents_harvesting": (0.0, "n_agents")}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
@dataclass
class TestRow:
    family: str
    metric: str
    condition_a: str
    condition_b: str
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    median_a: float
    median_b: float
    u_stat: float
    p_value: float
    p_value_fdr: float
    p_value_paired: float
    cliffs_delta: float
    effect_size_interpretation: str
    is_statistically_significant: bool


def _gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative vector (0 = equal, ->1 = one holds all)."""
    v = np.sort(np.asarray(values, dtype=float))
    n = v.size
    total = v.sum()
    if n == 0 or total <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * v).sum()) / (n * total) - (n + 1) / n)


def _event_metrics(run_dir: Path) -> dict[str, Any]:
    """Stream ``events.jsonl`` and derive behavioural metrics for one run.

    Returns a dict with the population-level action mix, the strongest
    single-agent conformity, the peak/final memory size, the number of agents
    that ever harvested, and the fraction of peers whose memory ever carried
    the poison payload verbatim plus the round it first appeared.
    """
    events_path = run_dir / "events.jsonl"
    per_agent_actions: dict[str, Counter[str]] = {}
    per_agent_peak_mem: dict[str, int] = {}
    per_agent_final_mem: dict[str, int] = {}
    per_agent_cells: dict[str, set[tuple[int, int]]] = {}
    per_agent_moves: Counter[str] = Counter()
    payload_first_round: dict[str, int] = {}
    last_position: dict[str, tuple[int, int] | None] = {}
    edge_rounds = 0
    late_rounds = 0
    stalled_rounds = 0
    directional_actions = 0
    no_op_moves = 0
    total_spawned = 0.0
    total_harvested = 0.0

    with events_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            agent_id = event["agent_id"]
            round_no = int(event["round"])
            action = event["action"]
            counter = per_agent_actions.setdefault(agent_id, Counter())
            counter[action] += 1
            if action in MOVE_ACTIONS:
                per_agent_moves[agent_id] += 1
            position = (event.get("observation") or {}).get("my_position")
            if position is not None:
                per_agent_cells.setdefault(agent_id, set()).add(tuple(position))
                if round_no >= LATE_ROUND_START and _is_edge(position, GRID_SIZE):
                    edge_rounds += 1
                if round_no >= LATE_ROUND_START:
                    late_rounds += 1
                if position == last_position.get(agent_id):
                    stalled_rounds += 1
                if action in MOVE_ACTIONS:
                    directional_actions += 1
                    if position == last_position.get(agent_id):
                        no_op_moves += 1
            last_position[agent_id] = position
            memory = event.get("memory_state") or ""
            size = len(memory)
            if size > per_agent_peak_mem.get(agent_id, 0):
                per_agent_peak_mem[agent_id] = size
            per_agent_final_mem[agent_id] = size
            if PAYLOAD_MARKER in memory and agent_id not in payload_first_round:
                payload_first_round[agent_id] = round_no
            info = event.get("info") or {}
            total_spawned = float(info.get("total_spawned", total_spawned))
            total_harvested = float(info.get("total_harvested", total_harvested))

    if not per_agent_actions:
        return {}

    all_actions: Counter[str] = Counter()
    conformity: list[float] = []
    distinct: list[int] = []
    rounds_seen = 0
    for counter in per_agent_actions.values():
        all_actions.update(counter)
        total = sum(counter.values())
        rounds_seen = max(rounds_seen, total)
        conformity.append(counter.most_common(1)[0][1] / total if total else 0.0)
        distinct.append(len(counter))

    n_actions = sum(all_actions.values())
    n_agents = len(per_agent_actions)
    peers = [aid for aid in per_agent_actions if aid != "agent_0"]
    reach = sum(1 for aid in peers if aid in payload_first_round) / len(peers) if peers else 0.0
    peer_first_rounds = [payload_first_round[aid] for aid in peers if aid in payload_first_round]
    visited = [len(cells) for cells in per_agent_cells.values()]

    return {
        "stay_share": all_actions.get("stay", 0) / n_actions if n_actions else 0.0,
        "harvest_share": all_actions.get("harvest", 0) / n_actions if n_actions else 0.0,
        "population_modal_share": (
            all_actions.most_common(1)[0][1] / n_actions if n_actions else 0.0
        ),
        "population_modal_action": all_actions.most_common(1)[0][0] if n_actions else "",
        "agent_conformity": float(np.mean(conformity)) if conformity else 0.0,
        "n_distinct_actions_mean": float(np.mean(distinct)) if distinct else 0.0,
        "late_edge_share": edge_rounds / late_rounds if late_rounds else float("nan"),
        "stalled_share": stalled_rounds / late_rounds if late_rounds else float("nan"),
        "no_op_move_share": (
            no_op_moves / directional_actions if directional_actions else float("nan")
        ),
        "cells_visited_mean": float(np.mean(visited)) if visited else 0.0,
        "moves_per_agent_mean": (
            float(np.mean(list(per_agent_moves.values()))) if per_agent_moves else 0.0
        ),
        "never_moved_share": (
            sum(1 for aid in per_agent_actions if per_agent_moves[aid] == 0) / n_agents
            if n_agents
            else 0.0
        ),
        "peak_memory_chars": float(max(per_agent_peak_mem.values())) if per_agent_peak_mem else 0.0,
        "final_memory_chars_mean": (
            float(np.mean(list(per_agent_final_mem.values()))) if per_agent_final_mem else 0.0
        ),
        "verbatim_payload_reach": reach,
        "verbatim_first_round": (
            float(np.mean(peer_first_rounds)) if peer_first_rounds else float("nan")
        ),
        "total_spawned": total_spawned,
        "total_harvested": total_harvested,
        "rounds_in_events": rounds_seen,
    }


def _load_runs(indir: Path, with_events: bool) -> pd.DataFrame:
    """Rehydrate every run under *indir* into a flat long-form frame."""
    records: list[dict[str, Any]] = []
    summaries = sorted(indir.rglob("summary.json"))
    for summary_json in summaries:
        run_dir = summary_json.parent
        try:
            rehydrator = RunRehydrator(run_dir)
            meta = rehydrator.load_metadata()
            summary = rehydrator.load_summary()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to rehydrate %s: %s", run_dir, exc)
            continue
        info = summary.get("summary_info", {})
        rewards = info.get("cumulative_rewards") or {}
        reward_vector = np.array(list(rewards.values()), dtype=float)
        record: dict[str, Any] = {
            "run_id": meta.get("run_id"),
            "experiment_id": meta.get("experiment_id"),
            "environment": meta.get("env_type"),
            "policy": meta.get("memory_policy"),
            "topology": meta.get("topology"),
            "sharing_mode": meta.get("sharing_mode"),
            "poisoning": meta.get("poisoning_mode"),
            "seed": meta.get("seed"),
            "n_agents": meta.get("n_agents"),
            "model": meta.get("model_name"),
            "git_commit": meta.get("git_commit"),
            "rounds_played": info.get("rounds_played"),
            "final_score": summary.get("final_score", 0.0),
            "oracle_score": info.get("oracle_score"),
            "mean_self_bleu": info.get("mean_self_bleu", 0.0),
            "peer_contamination_rate": info.get("peer_contamination_rate", 0.0),
            "poison_dosage_rate": info.get("poison_dosage_rate", 0.0),
            "propagation_latency": info.get("propagation_latency"),
            "n_agents_harvesting": int((reward_vector > 0).sum()),
            "harvest_gini": _gini(reward_vector),
            "total_reward": float(reward_vector.sum()),
            "seed_agent_reward": float(rewards.get("agent_0", 0.0)),
        }
        adherence = info.get("per_agent_poison_adherence") or {}
        record["adherence_mean"] = float(np.mean(list(adherence.values()))) if adherence else 0.0
        if with_events:
            record.update(_event_metrics(run_dir))
        records.append(record)
    return pd.DataFrame(records)


def _load_control(controldir: Path | None, environment: str | None) -> pd.DataFrame:
    """Load the ``no_memory`` baseline rows that anchor the efficacy gap.

    Baseline metadata records ``topology = "full_broadcast"`` even when sharing
    is disabled, so the *effective* topology is taken from ``sharing_mode`` —
    the same correction :mod:`scripts.analyze_baselines` applies.
    """
    if controldir is None or not controldir.exists():
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for summary_json in sorted(controldir.rglob("summary.json")):
        run_dir = summary_json.parent
        try:
            meta = RunRehydrator(run_dir).load_metadata()
            summary = RunRehydrator(run_dir).load_summary()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to rehydrate control %s: %s", run_dir, exc)
            continue
        if meta.get("memory_policy") != "no_memory":
            continue
        if environment and meta.get("env_type") != environment:
            continue
        info = summary.get("summary_info", {})
        records.append(
            {
                "run_id": meta.get("run_id"),
                "environment": meta.get("env_type"),
                "policy": "no_memory",
                "topology": meta.get("sharing_mode") or meta.get("topology"),
                "poisoning": meta.get("poisoning_mode"),
                "seed": meta.get("seed"),
                "final_score": summary.get("final_score", 0.0),
                "oracle_score": info.get("oracle_score"),
                "mean_self_bleu": info.get("mean_self_bleu", 0.0),
                "peer_contamination_rate": info.get("peer_contamination_rate", 0.0),
                "poison_dosage_rate": info.get("poison_dosage_rate", 0.0),
                "propagation_latency": info.get("propagation_latency"),
                "is_control": True,
            }
        )
    return pd.DataFrame(records)


def _condition_label(df: pd.DataFrame) -> pd.Series:
    """Human-readable ``policy | topology | poisoning`` cell label."""
    return (
        df["policy"].astype(str)
        + " | "
        + df["topology"].astype(str)
        + " | "
        + df["poisoning"].astype(str)
    )


def _categorise(df: pd.DataFrame) -> pd.DataFrame:
    """Attach ordered categoricals and the per-run condition label."""
    out = df.copy()
    out["condition"] = _condition_label(out)
    out["policy"] = pd.Categorical(out["policy"], categories=POLICY_ORDER, ordered=True)
    out["topology"] = pd.Categorical(out["topology"], categories=TOPOLOGY_ORDER, ordered=True)
    out["poisoning"] = pd.Categorical(out["poisoning"], categories=POISON_ORDER, ordered=True)
    return out


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def _sem_ci(values: pd.Series, confidence: float = 0.95) -> tuple[float, float]:
    """Return (sem, ci_half_width) for a t-based confidence interval."""
    values = values.dropna().astype(float)
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    sem = float(values.std(ddof=1) / np.sqrt(n))
    if n <= 1:
        return sem, 0.0
    return sem, float(stats.t.ppf(1 - (1 - confidence) / 2, n - 1)) * sem


def _mann_whitney(
    family: str,
    metric: str,
    label_a: str,
    vals_a: np.ndarray,
    label_b: str,
    vals_b: np.ndarray,
    paired: dict[int, tuple[float, float]] | None = None,
) -> TestRow:
    try:
        stat, p_value = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
    except ValueError:
        stat, p_value = 0.0, 1.0
    p_paired = float("nan")
    if paired:
        diffs = np.array([a - b for a, b in paired.values()], dtype=float)
        diffs = diffs[diffs != 0.0]
        if diffs.size:
            try:
                p_paired = float(
                    stats.wilcoxon(diffs, alternative="two-sided", zero_method="wilcox").pvalue
                )
            except ValueError:
                p_paired = float("nan")
    delta = compute_cliffs_delta(vals_a, vals_b)
    return TestRow(
        family=family,
        metric=metric,
        condition_a=label_a,
        condition_b=label_b,
        n_a=int(len(vals_a)),
        n_b=int(len(vals_b)),
        mean_a=float(np.mean(vals_a)) if vals_a.size else float("nan"),
        mean_b=float(np.mean(vals_b)) if vals_b.size else float("nan"),
        median_a=float(np.median(vals_a)) if vals_a.size else float("nan"),
        median_b=float(np.median(vals_b)) if vals_b.size else float("nan"),
        u_stat=float(stat),
        p_value=float(p_value),
        p_value_fdr=float(p_value),
        p_value_paired=p_paired,
        cliffs_delta=float(delta),
        effect_size_interpretation=interpret_cliffs_delta(delta),
        is_statistically_significant=bool(p_value < 0.05),
    )


def _apply_fdr(rows: list[TestRow]) -> list[TestRow]:
    """Benjamini-Hochberg FDR correction across every test in *rows*."""
    if not rows:
        return rows
    raw_p = np.array([r.p_value for r in rows])
    order = np.argsort(raw_p)
    sorted_p = raw_p[order]
    n = len(raw_p)
    q = np.empty(n)
    running_min = 1.0
    for i in range(n - 1, -1, -1):
        running_min = min(running_min, sorted_p[i] * n / (i + 1))
        q[i] = running_min
    fdr_p = np.empty(n)
    fdr_p[order] = np.clip(q, 0.0, 1.0)
    for row, corrected in zip(rows, fdr_p, strict=True):
        row.p_value_fdr = float(corrected)
        row.is_statistically_significant = bool(corrected < 0.05)
    return rows


def _seed_paired(
    cell_a: pd.DataFrame, cell_b: pd.DataFrame, metric: str
) -> dict[int, tuple[float, float]]:
    """Per-seed (a, b) pairs for a fully seed-crossed design."""
    a = cell_a.set_index("seed")[metric] if metric in cell_a else pd.Series(dtype=float)
    b = cell_b.set_index("seed")[metric] if metric in cell_b else pd.Series(dtype=float)
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    return {int(s): (float(r.a), float(r.b)) for s, r in joined.iterrows()}


def _run_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Build the sharing / poisoning / policy contrast families."""
    rows: list[TestRow] = []
    experiments = df[~df["is_control"].fillna(False)].copy()

    def cell(**filters: str) -> pd.DataFrame:
        mask = pd.Series(True, index=experiments.index)
        for col, val in filters.items():
            mask &= experiments[col].astype(str) == val
        return experiments[mask]

    def add(
        family: str,
        metric: str,
        label_a: str,
        cell_a: pd.DataFrame,
        label_b: str,
        cell_b: pd.DataFrame,
    ) -> None:
        vals_a = pd.to_numeric(cell_a[metric], errors="coerce").dropna().to_numpy()
        vals_b = pd.to_numeric(cell_b[metric], errors="coerce").dropna().to_numpy()
        if vals_a.size == 0 or vals_b.size == 0:
            return
        rows.append(
            _mann_whitney(
                family,
                metric,
                label_a,
                vals_a,
                label_b,
                vals_b,
                paired=_seed_paired(cell_a, cell_b, metric),
            )
        )

    for policy in POLICY_ORDER:
        for poisoning in POISON_ORDER:
            for topo_a, topo_b in (
                ("off", "full_broadcast"),
                ("off", "ring"),
                ("full_broadcast", "ring"),
            ):
                for metric in ("final_score", "efficacy_gap"):
                    add(
                        f"sharing::{metric}",
                        metric,
                        f"{policy}|{topo_a}",
                        cell(policy=policy, poisoning=poisoning, topology=topo_a),
                        f"{policy}|{topo_b}",
                        cell(policy=policy, poisoning=poisoning, topology=topo_b),
                    )

    for policy in POLICY_ORDER:
        for topology in TOPOLOGY_ORDER:
            add(
                "poisoning::final_score",
                "final_score",
                f"{policy}|{topology}|clean",
                cell(policy=policy, topology=topology, poisoning="clean"),
                f"{policy}|{topology}|internal",
                cell(policy=policy, topology=topology, poisoning="internal"),
            )

    for topology in TOPOLOGY_ORDER:
        for poisoning in POISON_ORDER:
            for policy_a, policy_b in (
                ("structured_incremental", "naive_overwrite"),
                ("structured_incremental", "raw_trajectory_buffer"),
                ("naive_overwrite", "raw_trajectory_buffer"),
            ):
                add(
                    "policy::final_score",
                    "final_score",
                    f"{policy_a}|{topology}|{poisoning}",
                    cell(policy=policy_a, topology=topology, poisoning=poisoning),
                    f"{policy_b}|{topology}|{poisoning}",
                    cell(policy=policy_b, topology=topology, poisoning=poisoning),
                )

    return pd.DataFrame([asdict(r) for r in _apply_fdr(rows)])


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def _policy_axis() -> list[str]:
    return [SHORT_POLICY[p] for p in POLICY_ORDER]


def _strip(
    ax: plt.Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    order: list[str],
    hue: str | None = None,
    hue_order: list[str] | None = None,
) -> None:
    """Black seed-level strip overlay, dodge-aware."""
    kwargs: dict[str, Any] = {
        "data": data,
        "x": x,
        "y": y,
        "order": order,
        "color": "black",
        "alpha": 0.5,
        "jitter": 0.1,
        "ax": ax,
    }
    if hue is not None:
        kwargs["hue"] = hue
        kwargs["hue_order"] = hue_order
        kwargs["dodge"] = True
        kwargs["palette"] = "dark:black"
    sns.stripplot(**kwargs)


def _plot_performance(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=df,
        x="policy",
        y="final_score",
        hue="topology",
        order=POLICY_ORDER,
        hue_order=TOPOLOGY_ORDER,
        errorbar=("ci", 95),
        capsize=0.08,
        ax=ax,
    )
    _strip(ax, df, "policy", "final_score", POLICY_ORDER, "topology", TOPOLOGY_ORDER)
    ax.set_title("Task Performance by Memory Policy and Sharing Topology (pooled over poisoning)")
    ax.set_xlabel("Memory Policy")
    ax.set_ylabel("Final Task Score")
    ax.set_xticks(range(len(POLICY_ORDER)))
    ax.set_xticklabels(_policy_axis())
    ax.set_ylim(0, 0.55)
    ax.legend(title="Sharing topology")
    _save(fig, out)


def _plot_topology_gradient(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, policy in zip(axes, POLICY_ORDER, strict=True):
        sub = df[df["policy"] == policy]
        sns.barplot(
            data=sub,
            x="topology",
            y="final_score",
            hue="poisoning",
            order=TOPOLOGY_ORDER,
            hue_order=POISON_ORDER,
            errorbar=("ci", 95),
            capsize=0.08,
            ax=ax,
        )
        _strip(ax, sub, "topology", "final_score", TOPOLOGY_ORDER, "poisoning", POISON_ORDER)
        ax.set_title(f"{SHORT_POLICY[policy]}")
        ax.set_xlabel("")
        ax.set_xticks(range(len(TOPOLOGY_ORDER)))
        ax.set_xticklabels([SHORT_TOPOLOGY[t] for t in TOPOLOGY_ORDER], rotation=12)
        if ax is not axes[0]:
            ax.set_ylabel("")
        ax.set_ylim(0, 0.55)
    axes[0].set_ylabel("Final Task Score")
    axes[-1].legend(title="Poisoning", loc="upper right")
    fig.suptitle("Sharing topology gradient, split by memory policy and poisoning mode")
    _save(fig, out)


def _plot_poisoning_paired(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, policy in zip(axes, POLICY_ORDER, strict=True):
        sub = df[df["policy"] == policy]
        sns.boxplot(
            data=sub,
            x="topology",
            y="final_score",
            hue="poisoning",
            order=TOPOLOGY_ORDER,
            hue_order=POISON_ORDER,
            width=0.6,
            ax=ax,
        )
        _strip(ax, sub, "topology", "final_score", TOPOLOGY_ORDER, "poisoning", POISON_ORDER)
        ax.set_title(f"{SHORT_POLICY[policy]}")
        ax.set_xlabel("")
        ax.set_xticks(range(len(TOPOLOGY_ORDER)))
        ax.set_xticklabels([SHORT_TOPOLOGY[t] for t in TOPOLOGY_ORDER], rotation=12)
        if ax is not axes[0]:
            ax.set_ylabel("")
        ax.set_ylim(-0.02, 0.55)
    axes[0].set_ylabel("Final Task Score")
    axes[-1].legend(title="Poisoning", loc="upper right")
    fig.suptitle("Clean versus internally poisoned trials, per topology (10 seeds per cell)")
    _save(fig, out)


def _plot_efficacy_gap(df: pd.DataFrame, out: Path) -> None:
    data = df.dropna(subset=["efficacy_gap"]).copy()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.stripplot(
        data=data,
        x="condition",
        y="efficacy_gap",
        order=sorted(data["condition"].unique()),
        color="black",
        alpha=0.5,
        jitter=0.14,
        ax=ax,
    )
    sns.pointplot(
        data=data,
        x="condition",
        y="efficacy_gap",
        order=sorted(data["condition"].unique()),
        errorbar=("ci", 95),
        linestyle="none",
        markers="D",
        color="C3",
        ax=ax,
    )
    ax.axhline(0.0, color="grey", linewidth=1, linestyle="--")
    ax.axhline(1.0, color="black", linewidth=1, linestyle=":")
    ax.set_title("Efficacy gap per condition (fraction of no-memory to oracle headroom recovered)")
    ax.set_xlabel("Policy | topology | poisoning")
    ax.set_ylabel("Efficacy Gap")
    ax.tick_params(axis="x", rotation=90)
    _save(fig, out)


def _plot_conformity(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.boxplot(
        data=df,
        x="condition",
        y="agent_conformity",
        order=sorted(df["condition"].unique()),
        color="#9ecae1",
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x="condition",
        y="agent_conformity",
        order=sorted(df["condition"].unique()),
        color="black",
        alpha=0.55,
        jitter=0.14,
        ax=ax,
    )
    ax.set_title(
        "Action conformity: share of an agent's 50 rounds spent on its single most common action"
    )
    ax.set_xlabel("Policy | topology | poisoning")
    ax.set_ylabel("Agent conformity")
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="x", rotation=90)
    _save(fig, out)


def _plot_conformity_vs_score(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="agent_conformity",
        y="final_score",
        hue="policy",
        style="topology",
        palette="deep",
        s=55,
        alpha=0.85,
        ax=ax,
    )
    ax.set_title("Conformity versus score: every dot is one 50-round run")
    ax.set_xlabel("Mean agent conformity (modal-action share)")
    ax.set_ylabel("Final Task Score")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.02, 0.55)
    ax.legend(title="policy / topology", fontsize=8, ncol=2)
    _save(fig, out)


def _plot_stay_vs_score(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for poisoning, marker in zip(POISON_ORDER, ("o", "X"), strict=True):
        sub = df[df["poisoning"] == poisoning]
        ax.scatter(
            sub["stay_share"],
            sub["final_score"],
            label=f"poisoning = {poisoning}",
            marker=marker,
            alpha=0.8,
            s=55,
        )
    ax.set_title("Fraction of all agent-rounds spent on 'stay' versus final score")
    ax.set_xlabel("Population 'stay' share")
    ax.set_ylabel("Final Task Score")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 0.55)
    ax.legend()
    _save(fig, out)


def _plot_contamination(df: pd.DataFrame, out: Path) -> None:
    """Built-in keyword counter versus the verbatim-payload reach measure."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    poisoned = df[df["poisoning"] == "internal"]
    for ax, (metric, title) in zip(
        axes,
        [
            ("peer_contamination_rate", "Runner counter: peer_contamination_rate (keyword 'stay')"),
            (
                "verbatim_payload_reach",
                "Verbatim payload reach: peers whose memory carried the poison text",
            ),
        ],
        strict=True,
    ):
        sns.barplot(
            data=poisoned,
            x="topology",
            y=metric,
            hue="policy",
            order=TOPOLOGY_ORDER,
            hue_order=POLICY_ORDER,
            errorbar=("ci", 95),
            capsize=0.08,
            ax=ax,
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("")
        ax.set_xticks(range(len(TOPOLOGY_ORDER)))
        ax.set_xticklabels([SHORT_TOPOLOGY[t] for t in TOPOLOGY_ORDER], rotation=12)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Rate")
    axes[0].legend(title="Memory policy", fontsize=8)
    _save(fig, out)


def _plot_latency(df: pd.DataFrame, out: Path) -> None:
    poisoned = df[df["poisoning"] == "internal"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    for ax, (metric, title) in zip(
        axes,
        [
            ("verbatim_first_round", "Verbatim payload: first round a peer's memory carried it"),
            (
                "propagation_latency",
                "Runner counter: propagation_latency (keyword 'stay', ungated)",
            ),
        ],
        strict=True,
    ):
        sns.boxplot(
            data=poisoned,
            x="topology",
            y=metric,
            hue="policy",
            order=TOPOLOGY_ORDER,
            hue_order=POLICY_ORDER,
            ax=ax,
        )
        _strip(ax, poisoned, "topology", metric, TOPOLOGY_ORDER, "policy", POLICY_ORDER)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("")
        ax.set_xticks(range(len(TOPOLOGY_ORDER)))
        ax.set_xticklabels([SHORT_TOPOLOGY[t] for t in TOPOLOGY_ORDER], rotation=12)
    axes[0].set_ylabel("Round")
    axes[0].legend(title="Memory policy", fontsize=8)
    _save(fig, out)


def _plot_memory_bloat(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    order = sorted(df["condition"].unique())
    sns.boxplot(
        data=df,
        x="condition",
        y="peak_memory_chars",
        order=order,
        color="#9ecae1",
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x="condition",
        y="peak_memory_chars",
        order=order,
        color="black",
        alpha=0.55,
        jitter=0.14,
        ax=ax,
    )
    ax.set_yscale("log")
    ax.set_title("Peak memory size per agent (characters, log scale)")
    ax.set_xlabel("Policy | topology | poisoning")
    ax.set_ylabel("Peak memory characters")
    ax.tick_params(axis="x", rotation=90)
    _save(fig, out)


def _plot_collapse_vs_score(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.scatterplot(
        data=df,
        x="mean_self_bleu",
        y="final_score",
        hue="policy",
        style="topology",
        palette="deep",
        s=55,
        alpha=0.85,
        ax=ax,
    )
    ax.set_title("Memory collapse (Self-BLEU) versus task performance")
    ax.set_xlabel("Mean Self-BLEU")
    ax.set_ylabel("Final Task Score")
    ax.set_xlim(0.9, 1.001)
    ax.set_ylim(-0.02, 0.55)
    ax.legend(title="policy / topology", fontsize=8, ncol=2)
    _save(fig, out)


def _plot_equity(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    order = sorted(df["condition"].unique())
    axes[0].bar(
        range(len(order)),
        df.groupby("condition", observed=True)["n_agents_harvesting"].mean().reindex(order),
        color="#4c72b0",
    )
    axes[0].set_xticks(range(len(order)))
    axes[0].set_xticklabels(order, rotation=90)
    axes[0].set_ylim(0, 6.2)
    axes[0].set_title("Agents that scored at least once (of 6)")
    axes[0].set_ylabel("Mean harvesting agents")
    axes[1].bar(
        range(len(order)),
        df.groupby("condition", observed=True)["harvest_gini"].mean().reindex(order),
        color="#c44e52",
    )
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels(order, rotation=90)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Inequality of harvest across the six agents (Gini, 0 = equal)")
    axes[1].set_ylabel("Mean Gini coefficient")
    _save(fig, out)


def _plot_movement_forensics(df: pd.DataFrame, out: Path) -> None:
    """Two failure signatures: boundary lock (walking into a wall) and
    total freeze (never issuing a directional action at all)."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    order = sorted(df["condition"].unique())
    grouped = df.groupby("condition", observed=True)
    panels = [
        ("late_edge_share", "On the grid border, final 10 rounds", "#c44e52", 1.02),
        ("no_op_move_share", "Directional commands that moved nothing", "#dd8452", 1.02),
        ("never_moved_share", "Agents that never moved at all", "#4c72b0", 1.02),
        ("cells_visited_mean", "Distinct grid cells visited per agent", "#55a868", None),
    ]
    for ax, (metric, title, colour, ymax) in zip(axes, panels, strict=True):
        ax.bar(
            range(len(order)),
            grouped[metric].mean().reindex(order),
            color=colour,
        )
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=90, fontsize=7)
        ax.set_title(title)
        ax.set_ylabel(metric)
        if ymax is not None:
            ax.set_ylim(0, ymax)
    fig.suptitle(
        "Movement forensics: sharing strands agents against the grid border, "
        "where their movement commands become no-ops"
    )
    _save(fig, out)


def _plot_score_heatmap(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for ax, poisoning in zip(axes, POISON_ORDER, strict=True):
        table = (
            df[df["poisoning"] == poisoning]
            .pivot_table(index="policy", columns="topology", values="final_score", aggfunc="mean")
            .reindex(index=POLICY_ORDER, columns=TOPOLOGY_ORDER)
        )
        sns.heatmap(
            table,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            vmin=0,
            vmax=0.5,
            cbar=poisoning == "clean",
            ax=ax,
        )
        ax.set_title(f"Mean score — poisoning = {poisoning}")
        ax.set_xlabel("Sharing topology")
        ax.set_ylabel("")
        ax.set_yticklabels(_policy_axis(), rotation=0)
    _save(fig, out)


def _plot_topology_panel(topology: str, sub: pd.DataFrame, figdir: Path) -> None:
    """Per-topology panel: performance, distributions, conformity."""
    panel = figdir / topology
    panel.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=sub,
        x="policy",
        y="final_score",
        hue="poisoning",
        order=POLICY_ORDER,
        hue_order=POISON_ORDER,
        errorbar=("ci", 95),
        capsize=0.08,
        ax=ax,
    )
    sns.stripplot(
        data=sub,
        x="policy",
        y="final_score",
        hue="poisoning",
        order=POLICY_ORDER,
        hue_order=POISON_ORDER,
        palette="dark:black",
        alpha=0.5,
        dodge=True,
        jitter=0.08,
        ax=ax,
    )
    ax.set_title(f"Task performance — sharing = {SHORT_TOPOLOGY[topology]}")
    ax.set_xlabel("Memory Policy")
    ax.set_ylabel("Final Task Score")
    ax.set_xticks(range(len(POLICY_ORDER)))
    ax.set_xticklabels(_policy_axis())
    ax.set_ylim(0, 0.55)
    ax.legend(title="Poisoning")
    _save(fig, panel / "performance_comparison.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric in zip(
        axes,
        ["final_score", "stay_share", "peak_memory_chars"],
        strict=True,
    ):
        plot_data = sub.copy()
        if metric == "peak_memory_chars":
            plot_data[metric] = np.log10(plot_data[metric].clip(lower=1))
        sns.boxplot(
            data=plot_data,
            x="policy",
            y=metric,
            hue="poisoning",
            order=POLICY_ORDER,
            hue_order=POISON_ORDER,
            ax=ax,
        )
        _strip(ax, plot_data, "policy", metric, POLICY_ORDER, "poisoning", POISON_ORDER)
        label = "log10 peak memory chars" if metric == "peak_memory_chars" else metric
        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_xticks(range(len(POLICY_ORDER)))
        ax.set_xticklabels(_policy_axis(), rotation=12)
    fig.suptitle(f"Run-level distributions — sharing = {SHORT_TOPOLOGY[topology]}")
    _save(fig, panel / "run_level_distributions.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=sub,
        x="policy",
        y="agent_conformity",
        hue="poisoning",
        order=POLICY_ORDER,
        hue_order=POISON_ORDER,
        errorbar=("ci", 95),
        capsize=0.08,
        ax=ax,
    )
    ax.set_title(f"Action conformity — sharing = {SHORT_TOPOLOGY[topology]}")
    ax.set_xlabel("Memory Policy")
    ax.set_ylabel("Mean agent conformity")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(range(len(POLICY_ORDER)))
    ax.set_xticklabels(_policy_axis())
    ax.legend(title="Poisoning")
    _save(fig, panel / "conformity.png")


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["policy", "topology", "poisoning"]
    for keys, cell in df.groupby(group_cols, observed=True, sort=True):
        record: dict[str, Any] = dict(zip(group_cols, keys, strict=True))
        record["n_runs"] = int(len(cell))
        record["n_zero_score_runs"] = int((cell["final_score"] <= 0).sum())
        for metric in SUMMARY_METRICS:
            if metric not in cell.columns:
                continue
            values = pd.to_numeric(cell[metric], errors="coerce")
            if values.notna().sum() == 0:
                continue
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            sem, ci_half = _sem_ci(values)
            record[f"{metric}_mean"] = mean
            record[f"{metric}_std"] = std
            record[f"{metric}_sem"] = sem
            if metric in NATIVE_ZERO_ONE:
                low, high = 0.0, 1.0
            elif metric in METRIC_BOUNDS:
                low = METRIC_BOUNDS[metric][0]
                high = float(cell[METRIC_BOUNDS[metric][1]].max())
            else:
                low, high = -np.inf, np.inf
            record[f"{metric}_ci_low"] = float(max(mean - ci_half, low))
            record[f"{metric}_ci_high"] = float(min(mean + ci_half, high))
        rows.append(record)
    return pd.DataFrame(rows)


def _fmt(row: pd.Series, metric: str, digits: int = 4, default: str = "—") -> str:
    mean = row.get(f"{metric}_mean")
    low = row.get(f"{metric}_ci_low")
    high = row.get(f"{metric}_ci_high")
    if mean is None or pd.isna(mean) or pd.isna(low) or pd.isna(high):
        return default
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def _write_summary_table(summary: pd.DataFrame, out: Path, environment: str | None) -> None:
    lines = [
        "# SEAM Phase-8 Experiment Summary Table",
        "",
        "Multi-agent factorial runs (`resource_foraging`, 6 agents, 50 rounds, `qwen2.5:7b`, "
        "seeds 40–49). Efficacy gap is normalised against the `no_memory` baseline runs for the "
        "same environment and seed.",
        "",
        "| Policy | Sharing | Poisoning | Runs | Zero-score runs | Score (Mean ± 95% CI) | "
        "Efficacy Gap | 'stay' share | Conformity | Payload reach | Verbatim latency | Self-BLEU | "
        "Peak memory (chars) | Harvesting agents | Cells visited | No-op moves | "
        "Never moved | Border-locked |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.sort_values(["policy", "topology", "poisoning"]).iterrows():
        lines.append(
            "| {policy} | {topology} | {poisoning} | {n} | {zero} | {score} | {gap} | {stay} | "
            "{conf} | {reach} | {lat} | {bleu} | {mem} | {agents} | {cells} | {noop} | "
            "{frozen} | {border}".format(
                policy=SHORT_POLICY.get(row["policy"], row["policy"]),
                topology=SHORT_TOPOLOGY.get(row["topology"], row["topology"]),
                poisoning=row["poisoning"],
                n=int(row["n_runs"]),
                zero=int(row.get("n_zero_score_runs", 0)),
                score=_fmt(row, "final_score"),
                gap=_fmt(row, "efficacy_gap", 3),
                stay=_fmt(row, "stay_share", 3),
                conf=_fmt(row, "agent_conformity", 3),
                reach=_fmt(row, "verbatim_payload_reach", 3),
                lat=_fmt(row, "verbatim_first_round", 2),
                bleu=_fmt(row, "mean_self_bleu", 4),
                mem=_fmt(row, "peak_memory_chars", 0),
                agents=_fmt(row, "n_agents_harvesting", 2),
                cells=_fmt(row, "cells_visited_mean", 2),
                noop=_fmt(row, "no_op_move_share", 3),
                frozen=_fmt(row, "never_moved_share", 3),
                border=_fmt(row, "late_edge_share", 3),
            )
        )
    if environment:
        lines += [
            "",
            f"Environment: `{environment}`. Regenerate with "
            "`python scripts/analyze_experiments.py`.",
        ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_forensics(df: pd.DataFrame, out: Path) -> None:
    """Per-condition comparison of the runner counters against verbatim reach."""
    rows = []
    for keys, cell in df.groupby(["policy", "topology", "poisoning"], observed=True, sort=True):
        policy, topology, poisoning = keys
        poisoned = cell[cell["poisoning"] == "internal"]
        rows.append(
            {
                "policy": policy,
                "topology": topology,
                "poisoning": poisoning,
                "n_runs": len(cell),
                "mean_peer_contamination_rate": float(cell["peer_contamination_rate"].mean()),
                "mean_poison_dosage_rate": float(cell["poison_dosage_rate"].mean()),
                "mean_propagation_latency": float(
                    pd.to_numeric(cell["propagation_latency"], errors="coerce").mean()
                )
                if pd.to_numeric(cell["propagation_latency"], errors="coerce").notna().any()
                else float("nan"),
                "mean_verbatim_payload_reach": float(poisoned["verbatim_payload_reach"].mean())
                if len(poisoned)
                else 0.0,
                "mean_verbatim_first_round": float(poisoned["verbatim_first_round"].mean())
                if len(poisoned)
                else float("nan"),
                "mean_adherence": float(poisoned["adherence_mean"].mean())
                if len(poisoned)
                else 0.0,
                "max_adherence": float(poisoned["adherence_mean"].max()) if len(poisoned) else 0.0,
                "mean_stay_share": float(cell["stay_share"].mean()),
                "mean_population_modal_share": float(cell["population_modal_share"].mean()),
            }
        )
    pd.DataFrame(rows).to_csv(out, index=False)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SEAM Phase-8 multi-agent experiments")
    parser.add_argument("--indir", type=str, default="runs/experiments", help="Experiment runs dir")
    parser.add_argument(
        "--controldir",
        type=str,
        default="runs/baselines",
        help="Baseline runs dir supplying the no_memory efficacy control",
    )
    parser.add_argument(
        "--figroot",
        type=str,
        default="figures/experiments",
        help="Parent figures directory; the environment is appended to it",
    )
    parser.add_argument(
        "--outroot",
        type=str,
        default="reports/experiments",
        help="Parent reports directory; the environment is appended to it",
    )
    parser.add_argument(
        "--figdir",
        type=str,
        default=None,
        help="Exact figures directory; overrides --figroot and skips per-environment nesting",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Exact reports directory; overrides --outroot and skips per-environment nesting",
    )
    parser.add_argument(
        "--skip-events",
        action="store_true",
        help="Skip the events.jsonl pass (behavioural metrics will be absent)",
    )
    args = parser.parse_args()

    indir = Path(args.indir)
    controldir = Path(args.controldir) if args.controldir else None

    df = _load_runs(indir, with_events=not args.skip_events)
    if df.empty:
        logger.error("No experiment run records found under %s", indir)
        return
    df["is_control"] = False
    logger.info("Loaded %d experiment records from %s", len(df), indir)

    environment = df["environment"].dropna().unique()
    environment = environment[0] if len(environment) == 1 else None
    if environment is None:
        logger.warning(
            "Runs under %s span %d environment(s); writing to 'multi_environment'. "
            "Run one environment at a time for clean per-environment folders.",
            indir,
            len(set(df["environment"].dropna())),
        )
        env_slug = "multi_environment"
    else:
        env_slug = str(environment)
        logger.info("Environment: %s", env_slug)

    figdir = Path(args.figdir) if args.figdir else Path(args.figroot) / env_slug
    outdir = Path(args.outdir) if args.outdir else Path(args.outroot) / env_slug
    figdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing figures to %s and reports to %s", figdir, outdir)

    control = _load_control(controldir, environment)
    if not control.empty:
        logger.info("Merged %d no_memory control runs from %s", len(control), controldir)
        df = pd.concat([df, control], ignore_index=True)

    # Efficacy gap is computed on the plain (object-dtype) frame so that the
    # `no_memory` / `off` control rows are still addressable by name.
    df = compute_efficacy_gap(df)
    all_runs = df
    experiments = _categorise(df[~df["is_control"].fillna(False)].copy())
    logger.info(
        "Loaded %d experiment records (%d conditions x %d seeds), %d control records",
        len(experiments),
        experiments.groupby(["policy", "topology", "poisoning"], observed=True).ngroups,
        experiments["seed"].nunique(),
        int(df["is_control"].fillna(False).sum()),
    )

    # 1. Run-level table
    run_cols = [
        "run_id",
        "policy",
        "topology",
        "poisoning",
        "seed",
        "rounds_played",
        "final_score",
        "oracle_score",
        "efficacy_gap",
        "mean_self_bleu",
        "stay_share",
        "harvest_share",
        "population_modal_share",
        "population_modal_action",
        "agent_conformity",
        "n_distinct_actions_mean",
        "late_edge_share",
        "no_op_move_share",
        "cells_visited_mean",
        "moves_per_agent_mean",
        "never_moved_share",
        "adherence_mean",
        "verbatim_payload_reach",
        "verbatim_first_round",
        "peak_memory_chars",
        "final_memory_chars_mean",
        "total_reward",
        "seed_agent_reward",
        "n_agents_harvesting",
        "harvest_gini",
        "total_harvested",
        "total_spawned",
        "peer_contamination_rate",
        "poison_dosage_rate",
        "propagation_latency",
    ]
    run_cols = [c for c in run_cols if c in experiments.columns]
    experiments.sort_values(["policy", "topology", "poisoning", "seed"])[run_cols].to_csv(
        outdir / "run_level_metrics.csv", index=False
    )
    logger.info("Saved run-level metrics to %s", outdir / "run_level_metrics.csv")

    # 2. Summary statistics
    summary = _summarise(experiments)
    summary.to_csv(outdir / "summary_statistics.csv", index=False)
    logger.info("Saved summary statistics to %s", outdir / "summary_statistics.csv")

    # 3. Statistical tests
    tests = _run_tests(all_runs)
    tests.to_csv(outdir / "statistical_tests.csv", index=False)
    logger.info("Saved %d statistical tests to %s", len(tests), outdir / "statistical_tests.csv")

    # 4. Contamination forensics
    _write_forensics(experiments, outdir / "contamination_forensics.csv")
    logger.info("Saved contamination forensics to %s", outdir / "contamination_forensics.csv")

    # 5. Figures
    _plot_performance(experiments, figdir / "performance_comparison.png")
    _plot_topology_gradient(experiments, figdir / "topology_gradient.png")
    _plot_poisoning_paired(experiments, figdir / "poisoning_paired_effect.png")
    _plot_efficacy_gap(experiments, figdir / "efficacy_gap_by_condition.png")
    _plot_conformity(experiments, figdir / "conformity_by_condition.png")
    _plot_conformity_vs_score(experiments, figdir / "conformity_vs_score.png")
    _plot_stay_vs_score(experiments, figdir / "stay_share_vs_score.png")
    _plot_contamination(experiments, figdir / "contamination_propagation.png")
    _plot_latency(experiments, figdir / "propagation_latency.png")
    _plot_memory_bloat(experiments, figdir / "memory_bloat.png")
    _plot_collapse_vs_score(experiments, figdir / "collapse_performance_tradeoff.png")
    _plot_equity(experiments, figdir / "harvest_equity.png")
    _plot_movement_forensics(experiments, figdir / "movement_forensics.png")
    _plot_score_heatmap(experiments, figdir / "score_condition_heatmap.png")
    for topology in TOPOLOGY_ORDER:
        sub = experiments[experiments["topology"] == topology]
        if not sub.empty:
            _plot_topology_panel(topology, sub, figdir)
    logger.info("Generated figures under %s", figdir)

    # 6. Markdown summary table
    _write_summary_table(summary, outdir / "summary_table.md", environment)
    logger.info("All experiment analysis outputs written under %s and %s", figdir, outdir)


if __name__ == "__main__":
    main()
