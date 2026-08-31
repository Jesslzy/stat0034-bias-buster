"""Measure agreement between human and automated EEDI scores."""

import argparse
import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

logger = logging.getLogger(__name__)

_HUMAN_SCORE_COL = "Overall EEDI risk score (1-5)"


def add_parser(subparsers) -> None:
    """Register the `validate` subcommand and its arguments.

    Args:
        subparsers: Subparser action from the parent ArgumentParser.
    """
    p = subparsers.add_parser("validate", help="Compute human vs. LLM judge agreement on the annotation sample.")
    p.add_argument(
        "--annotations",
        default="data/annotation_sample_blind.xlsx",
        help="Completed blind-annotation workbook (Annotations sheet).",
    )
    p.add_argument(
        "--reference",
        default="data/annotation_sample.csv",
        help="Full reference sample with the judge's own eedi_score.",
    )


def run(args: argparse.Namespace) -> None:
    """Compute and log human vs. LLM judge agreement statistics.

    Args:
        args: Parsed CLI arguments from add_parser.
    """
    human = pd.read_excel(args.annotations, sheet_name="Annotations")
    human = human[["item_id", _HUMAN_SCORE_COL]].rename(columns={_HUMAN_SCORE_COL: "human_score"})
    human = human.dropna(subset=["human_score"])

    judge = pd.read_csv(args.reference)[["item_id", "eedi_score"]].rename(columns={"eedi_score": "judge_score"})

    agreement = human.merge(judge, on="item_id", how="inner")
    logger.info("%d annotated items matched against %d judge reference rows", len(agreement), len(judge))

    h = agreement["human_score"].astype(int).to_numpy()
    j = agreement["judge_score"].astype(int).to_numpy()

    confusion = pd.crosstab(agreement["human_score"], agreement["judge_score"], rownames=["Human"], colnames=["Judge"])
    logger.info("Confusion matrix:\n%s", confusion)

    logger.info("Exact agreement:          %.1f%%", 100 * np.mean(h == j))
    logger.info("Within one point:         %.1f%%", 100 * np.mean(np.abs(h - j) <= 1))
    logger.info("Spearman correlation:     %.3f", spearmanr(h, j).correlation)
    logger.info("Unweighted Cohen's kappa: %.3f", cohen_kappa_score(h, j))
    logger.info("Quadratic-weighted kappa: %.3f", cohen_kappa_score(h, j, weights="quadratic"))
