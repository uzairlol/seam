"""CLI script to execute Phase 5 single-agent baseline runs across memory policies."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from seam.orchestration.config_loader import (
    EnvConfig,
    ExperimentConfig,
    MemoryConfig,
    ModelConfig,
    PoisoningConfig,
    SharingConfig,
)
from seam.orchestration.runner import EpisodeRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_baselines(
    env_type: str = "resource_foraging",
    model_name: str = "qwen2.5:7b",
    seeds: list[int] | None = None,
    output_dir: str = "runs/baselines",
) -> list[dict]:
    """Execute baseline runs for all 3 memory policies without sharing.

    Args:
        env_type: Target environment type.
        model_name: Name of the model in Ollama.
        seeds: List of random seeds to evaluate.
        output_dir: Parent output folder.

    Returns:
        List of summary result dictionaries.
    """
    target_seeds = seeds or [1, 2, 3]
    policies = ["naive_overwrite", "raw_trajectory_buffer", "structured_incremental"]
    results = []

    for policy in policies:
        logger.info("==========================================")
        logger.info("Running Baseline Policy: %s (model=%s)", policy, model_name)
        logger.info("==========================================")

        cfg = ExperimentConfig(
            experiment_id=f"baseline_{policy}",
            description=f"Phase 5 single-agent baseline run for {policy}",
            env=EnvConfig(type=env_type, n_agents=4, episode_length=20),
            model=ModelConfig(model_name=model_name, base_url="http://localhost:11434"),
            memory=MemoryConfig(policy=policy),
            sharing=SharingConfig(mode="off"),
            poisoning=PoisoningConfig(mode="clean"),
            seeds=target_seeds,
        )

        for seed in target_seeds:
            logger.info("Executing seed %d for %s ...", seed, policy)
            runner = EpisodeRunner(config=cfg, seed=seed, base_dir=output_dir)
            summary = runner.run()
            results.append(summary)
            logger.info(
                "Completed %s (seed %d) — Score: %.4f, Mean Self-BLEU: %.4f",
                policy,
                seed,
                summary["final_score"],
                summary["mean_self_bleu"],
            )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SEAM Phase 5 Baselines")
    parser.add_argument("--env", type=str, default="resource_foraging", help="Environment type")
    parser.add_argument("--model", type=str, default="qwen2.5:7b", help="Ollama model tag")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3], help="Random seeds")
    parser.add_argument("--outdir", type=str, default="runs/baselines", help="Output directory")
    args = parser.parse_args()

    run_baselines(env_type=args.env, model_name=args.model, seeds=args.seeds, output_dir=args.outdir)


if __name__ == "__main__":
    main()

