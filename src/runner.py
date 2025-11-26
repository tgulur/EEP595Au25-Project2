"""Experiment runner (single, testable implementation).

This module provides a thin orchestration layer that wires together an
experiment class with a command-line interface. The previous implementation
contained duplicated code paths which made testing and coverage brittle; this
file consolidates the behavior into a single `run()` entrypoint that's easy to
call from tests.
"""
from __future__ import annotations

import argparse
import logging
from typing import Optional, Type

logger = logging.getLogger(__name__)


def run(experiment_cls: Type, argv: Optional[list] = None) -> int:
    """Run the experiment using the provided Experiment class.

    Args:
        experiment_cls: Class implementing the experiment (constructor: config_path, seed)
        argv: Optional list of CLI arguments (for testing); if None uses sys.argv

    Returns:
        Exit code (0 success, non-zero on error)
    """
    parser = argparse.ArgumentParser(description="CAN IDS Experiment (runner)")
    parser.add_argument("--config", default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed override")
    parser.add_argument("--quiet", action="store_true", help="Set logging to WARNING level")
    args = parser.parse_args(argv)

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    try:
        logger.info("Starting experiment runner")
        experiment = experiment_cls(config_path=args.config, seed=args.seed)
        experiment.run_experiment()
        logger.info("Experiment finished successfully")
        return 0
    except Exception as e:
        logger.exception("Experiment failed: %s", e)
        return 2


def main():
    # Helper for running this module directly. Importing the top-level
    # experiment class is deferred to avoid circular imports during testing.
    try:
        from main_experiment import CANIDSExperiment

        raise SystemExit(run(CANIDSExperiment))
    except Exception:
        logger.exception("Failed to run runner as script")
        raise


if __name__ == "__main__":
    main()
