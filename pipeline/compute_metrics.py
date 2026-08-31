"""Compute linguistic metrics for generated letters."""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from constants import REFERENCE_AGE, REFERENCE_GENDER, REFERENCE_TENURE
from metrics.agency import agency_ratio
from metrics.framing import gain_loss_ratio

logger = logging.getLogger(__name__)


def _get_reference_text(group: pd.DataFrame, stage: str) -> str | None:
    """Return the response text of the first replicate of the reference condition.

    Args:
        group: Subset of the outputs DataFrame already filtered to one (platform, stage).
        stage: Stage identifier ("stage1" or "stage2") to filter by.

    Returns:
        Response text of the reference row, or None if not found.
    """
    mask = (group["gender"] == REFERENCE_GENDER) & (group["age_group"] == REFERENCE_AGE)
    if stage == "stage2":
        mask &= group["tenure"] == REFERENCE_TENURE

    ref_rows = group[mask].sort_values("replicate")
    if ref_rows.empty:
        return None
    return ref_rows.iloc[0]["response_text"]


def _compute_semantic_by_group(df: pd.DataFrame) -> pd.Series:
    """Compute semantic distance from the reference condition per (platform, stage) group.

    Args:
        df: DataFrame containing at least platform, stage, and response_text columns.

    Returns:
        Series of cosine distances aligned to df's index. NaN where reference is missing.
    """
    # Deferred: sentence-transformers pulls in torch, which takes several
    # seconds to import — only pay that cost when semantic scoring runs.
    logger.info("Loading metrics.semantic module ...")
    t0 = time.perf_counter()
    from metrics.semantic import compute_semantic_distances

    logger.info(" module import took %.1fs", time.perf_counter() - t0)

    distances = pd.Series(index=df.index, dtype=float)

    for (platform, stage), group in df.groupby(["platform", "stage"]):
        ref_text = _get_reference_text(group, stage)
        if ref_text is None:
            logger.warning(
                "No reference output found for platform=%s stage=%s — " "semantic_distance will be NaN for this group.",
                platform,
                stage,
            )
            distances.loc[group.index] = float("nan")
            continue

        texts = group["response_text"].tolist()
        dists = compute_semantic_distances(texts, reference_text=ref_text)
        distances.loc[group.index] = dists
        logger.info("  %s | %s: %d rows scored", platform, stage, len(group))

    return distances


def add_parser(subparsers) -> None:
    """Register the `score` subcommand and its arguments.

    Args:
        subparsers: Subparser action from the parent ArgumentParser.
    """
    p = subparsers.add_parser("score", help="Compute linguistic metrics on collected outputs.")
    p.add_argument(
        "--in",
        dest="input",
        default=None,
        help="Input CSV. Defaults to data/raw_outputs.csv, or data/scored_outputs.csv "
        "when --only is set (so the other already-computed metric columns are "
        "available to carry through untouched).",
    )
    p.add_argument(
        "--out",
        dest="output",
        default=None,
        help="Output CSV. Defaults to data/scored_outputs.csv, updating that file " "in place when --only is set.",
    )
    p.add_argument(
        "--only",
        choices=["agency", "formality", "framing", "semantic"],
        default=None,
        help="Only (re)compute this one metric, leaving any other existing metric "
        "columns in --in untouched. Omit to compute all four (default).",
    )


def run(args: argparse.Namespace) -> None:
    """Compute the requested linguistic metric(s) and write the scored CSV.

    Args:
        args: Parsed CLI arguments from add_parser.
    """
    default_in = "data/scored_outputs.csv" if args.only else "data/raw_outputs.csv"
    in_path = Path(args.input or default_in)
    out_path = Path(args.output or "data/scored_outputs.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    logger.info("Loaded %d rows from %s", len(df), in_path)

    compute_all = args.only is None

    if compute_all or args.only == "agency":
        logger.info("Computing agency ratio...")
        df["agency_ratio"] = df["response_text"].map(agency_ratio)

    if compute_all or args.only == "formality":
        logger.info("Computing formality score...")
        # Deferred: spaCy takes several seconds to import — only pay that cost
        # when formality scoring runs.
        logger.info("Loading metrics.formality module ...")
        t0 = time.perf_counter()
        from metrics.formality import formality_scores

        logger.info("  module import took %.1fs", time.perf_counter() - t0)
        df["formality_score"] = formality_scores(df["response_text"].tolist())

    if compute_all or args.only == "framing":
        logger.info("Computing gain/loss framing ratio...")
        df["gain_loss_ratio"] = df["response_text"].map(gain_loss_ratio)

    if compute_all or args.only == "semantic":
        logger.info("Computing semantic distances...")
        df["semantic_distance"] = _compute_semantic_by_group(df)

    df.to_csv(out_path, index=False)
    logger.info("Saved %d scored rows to %s", len(df), out_path)
