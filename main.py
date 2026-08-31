"""Provide the command-line interface for the EEDI auditing pipeline."""

import argparse
import logging

from dotenv import load_dotenv

load_dotenv()

from pipeline import (
    compute_metrics,
    extract_breed,
    llm_judge,
    run_experiments,
    sample_for_annotation,
    validate_annotations,
)
from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for the Bias Busters CLI."""
    setup_logging()

    parser = argparse.ArgumentParser(
        prog="bias-busters",
        description="EEDI auditing pipeline for LLM-generated fundraising content.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_experiments.add_parser(subparsers)
    compute_metrics.add_parser(subparsers)
    extract_breed.add_parser(subparsers)
    llm_judge.add_parser(subparsers)
    sample_for_annotation.add_parser(subparsers)
    validate_annotations.add_parser(subparsers)

    args = parser.parse_args()

    if args.command == "collect":
        logger.info("Run collect data pipeline")
        run_experiments.run(args)
    elif args.command == "score":
        logger.info("Run scoring pipeline")
        compute_metrics.run(args)
    elif args.command == "extract-breed":
        logger.info("Run breed extraction pipeline")
        extract_breed.run(args)
    elif args.command == "judge":
        logger.info("Run LLM Judge pipeline")
        llm_judge.run(args)
    elif args.command == "sample":
        logger.info("Run annotation sampling pipeline")
        sample_for_annotation.run(args)
    elif args.command == "validate":
        logger.info("Run human vs. LLM judge agreement validation")
        validate_annotations.run(args)


if __name__ == "__main__":
    main()
