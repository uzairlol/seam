"""CLI script to execute Phase 8 full factorial multi-agent experiment matrices."""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from seam.orchestration.config_loader import (
    EnvConfig,
    ExperimentConfig,
    MemoryConfig,
    ModelConfig,
    PoisoningConfig,
    SharingConfig,
)
from seam.orchestration.runner import EpisodeRunner
from seam.utils.io import save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_experiment_grid(
    policies: list[str],
    sharing_modes: list[str],
    poisoning_modes: list[str],
    seeds: list[int],
) -> list[dict[str, Any]]:
    """Generate list of condition dicts covering all grid combinations.

    Args:
        policies: List of memory policy names.
        sharing_modes: List of sharing mode/topology names.
        poisoning_modes: List of poisoning mode names.
        seeds: List of random seeds.

    Returns:
        List of dicts representing experiment parameter combinations.
    """
    grid = []
    for policy in policies:
        for sharing in sharing_modes:
            topology = "off" if sharing == "off" else ("full_broadcast" if sharing in ("broadcast", "full_broadcast") else sharing)
            mode_str = "off" if sharing == "off" else "broadcast"

            for poisoning in poisoning_modes:
                for seed in seeds:
                    grid.append({
                        "policy": policy,
                        "sharing_mode": mode_str,
                        "topology": topology,
                        "poisoning_mode": poisoning,
                        "seed": seed,
                    })
    return grid


def run_experiments(
    env_type: str = "resource_foraging",
    model_name: str = "qwen2.5:7b",
    policies: list[str] | None = None,
    sharing_modes: list[str] | None = None,
    poisoning_modes: list[str] | None = None,
    seeds: list[int] | None = None,
    output_dir: str = "runs/experiments",
    resume: bool = True,
) -> list[dict[str, Any]]:
    """Execute multi-agent experiments across all requested grid conditions.

    Args:
        env_type: Target environment type.
        model_name: Ollama model tag.
        policies: Memory policy list.
        sharing_modes: Sharing topology/mode list.
        poisoning_modes: Poisoning mode list.
        seeds: List of random seeds.
        output_dir: Output base folder.
        resume: If True, skip runs that already have a summary.json.

    Returns:
        List of summary result dictionaries.
    """
    target_policies = policies or ["naive_overwrite", "raw_trajectory_buffer", "structured_incremental"]
    target_sharing = sharing_modes or ["off", "full_broadcast", "ring"]
    target_poisoning = poisoning_modes or ["clean", "internal"]
    target_seeds = seeds or [42, 43, 44]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    grid = generate_experiment_grid(
        policies=target_policies,
        sharing_modes=target_sharing,
        poisoning_modes=target_poisoning,
        seeds=target_seeds,
    )

    logger.info("==========================================")
    logger.info("Starting Phase 8 Experiment Grid Execution")
    logger.info("Total Conditions: %d (env=%s, model=%s)", len(grid), env_type, model_name)
    logger.info("==========================================")

    results: list[dict[str, Any]] = []

    for idx, cond in enumerate(grid, start=1):
        exp_id = f"exp_{cond['policy']}_{cond['topology']}_{cond['poisoning_mode']}"
        logger.info(
            "[%d/%d] Running %s | seed=%d | policy=%s | topology=%s | poison=%s",
            idx,
            len(grid),
            exp_id,
            cond["seed"],
            cond["policy"],
            cond["topology"],
            cond["poisoning_mode"],
        )

        cfg = ExperimentConfig(
            experiment_id=exp_id,
            description=f"Phase 8 multi-agent experiment run for {exp_id}",
            env=EnvConfig(type=env_type, n_agents=4, episode_length=20),
            model=ModelConfig(model_name=model_name, base_url="http://localhost:11434"),
            memory=MemoryConfig(policy=cond["policy"]),
            sharing=SharingConfig(
                mode=cond["sharing_mode"],
                topology=cond["topology"],
                publish_every_n_rounds=2,
            ),
            poisoning=PoisoningConfig(
                mode=cond["poisoning_mode"],
                poison_agent_id="agent_0",
            ),
            seeds=[cond["seed"]],
        )

        # Execute using EpisodeRunner context manager for strict memory safety
        with EpisodeRunner(config=cfg, seed=cond["seed"], base_dir=out_path) as runner:
            summary = runner.run()

        del runner
        gc.collect()

        summary["experiment_id"] = exp_id
        summary["policy"] = cond["policy"]
        summary["topology"] = cond["topology"]
        summary["poisoning_mode"] = cond["poisoning_mode"]
        results.append(summary)

    # Export aggregated results summary CSV and manifest JSON
    _save_summary_manifest(results, out_path)
    return results


def _save_summary_manifest(results: list[dict[str, Any]], out_dir: Path) -> None:
    """Save results_summary.csv and manifest.json to out_dir."""
    if not results:
        return

    rows = []
    for r in results:
        rows.append({
            "run_id": r.get("run_id"),
            "experiment_id": r.get("experiment_id"),
            "policy": r.get("policy"),
            "topology": r.get("topology"),
            "poisoning_mode": r.get("poisoning_mode"),
            "seed": r.get("seed"),
            "rounds_played": r.get("rounds_played"),
            "final_score": r.get("final_score"),
            "mean_self_bleu": r.get("mean_self_bleu"),
            "peer_contamination_rate": r.get("peer_contamination_rate", 0.0),
        })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "results_summary.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved aggregated summary CSV to %s", csv_path)

    manifest_path = out_dir / "manifest.json"
    save_json({"total_runs": len(results), "results": rows}, manifest_path)
    logger.info("Saved manifest JSON to %s", manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SEAM Phase 8 Multi-Agent Factorial Experiments")
    parser.add_argument("--env", type=str, default="resource_foraging", help="Environment type")
    parser.add_argument("--model", type=str, default="qwen2.5:7b", help="Ollama model tag")
    parser.add_argument("--policies", type=str, nargs="+", default=["naive_overwrite", "raw_trajectory_buffer", "structured_incremental"], help="Memory policies")
    parser.add_argument("--sharing", type=str, nargs="+", default=["off", "full_broadcast", "ring"], help="Sharing topologies")
    parser.add_argument("--poisoning", type=str, nargs="+", default=["clean", "internal"], help="Poisoning modes")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Random seeds")
    parser.add_argument("--outdir", type=str, default="runs/experiments", help="Output directory")
    args = parser.parse_args()

    run_experiments(
        env_type=args.env,
        model_name=args.model,
        policies=args.policies,
        sharing_modes=args.sharing,
        poisoning_modes=args.poisoning,
        seeds=args.seeds,
        output_dir=args.outdir,
    )


if __name__ == "__main__":
    main()
