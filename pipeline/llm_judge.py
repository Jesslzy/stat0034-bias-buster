"""Assign structured EEDI risk scores to generated letters."""

import argparse
import asyncio
import logging
import time
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from constants import DEFAULT_JUDGE_MODEL, JUDGE_CHECKPOINT_EVERY, JUDGE_CONCURRENCY
from models.models import JudgeOutput
from utils.llm_client import acall_with_retry
from utils.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

_KEY_COLS = ["condition_id", "platform", "replicate"]
_JUDGE_COLS = ["eedi_score", "flagged_dimension", "eedi_justification", "judge_raw_response"]


def _minmax_scale(series: pd.Series) -> pd.Series:
    """Scale a metric to the zero-to-one range.

    Args:
        series: Metric values to scale.

    Returns:
        Scaled values, or zeros when the metric is constant.
    """
    value_range = series.max() - series.min()
    return (series - series.min()) / value_range if value_range > 0 else series * 0


def _validate_columns(df: pd.DataFrame) -> None:
    """Raise if any required columns are missing from the input DataFrame.

    Args:
        df: Input DataFrame to validate.

    Raises:
        ValueError: If any required columns are absent.
    """
    required = {
        "agency_ratio",
        "formality_score",
        "gain_loss_ratio",
        "semantic_distance",
        "response_text",
        "prompt_text",
        "platform",
        "stage",
        "condition_id",
        "replicate",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input csv is missing columns: {missing}. Run `score` first.")


async def call_judge(model: str, system: str, user: str, temperature: float | None = None) -> JudgeOutput | None:
    """Call the judge model with tenacity-managed exponential back-off retry.

    Args:
        model: litellm model string for the judge.
        system: System prompt text.
        user: User prompt text for this row.
        temperature: temperature for the judge call.

    Returns:
        Parsed JudgeOutput, or None if all attempts fail, an auth error
        occurs, or the response doesn't validate against the schema.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    kwargs = {"max_tokens": 4000, "response_format": JudgeOutput}
    if temperature is not None:
        kwargs["temperature"] = temperature

    raw = await acall_with_retry(model, messages, **kwargs)
    if raw is None:
        return None
    try:
        return JudgeOutput.model_validate_json(raw)
    except ValidationError as exc:
        logger.warning("Judge response failed schema validation: %s\n%s", exc, raw)
        return None


async def _judge_one(
    position: int,
    idx: int,
    df: pd.DataFrame,
    model: str,
    system_prompt: str,
    user_template: str,
    temperature: float | None,
    semaphore: asyncio.Semaphore,
) -> tuple[int, int, pd.Series, JudgeOutput | None]:
    """Judge one row without mutating shared state.

    Args:
        position: Position of the row in the pending work.
        idx: DataFrame index of the row to process.
        df: Source data containing the generated letter and metrics.
        model: LiteLLM model identifier used by the judge.
        system_prompt: Judge system prompt.
        user_template: Judge user prompt template.
        temperature: Sampling temperature, or ``None`` for the model default.
        semaphore: Limit on concurrent judge calls.

    Returns:
        The work position, row index, source row, and judge result.
    """
    row = df.loc[idx]
    user_prompt = user_template.format(
        response_text=row["response_text"],
        agency_ratio=row["agency_ratio"],
        formality_score=row["formality_score"],
        gain_loss_ratio=row["gain_loss_ratio"],
        semantic_distance=row["semantic_distance"],
    )
    async with semaphore:
        result = await call_judge(model, system_prompt, user_prompt, temperature)
    return position, idx, row, result


async def _run_judging(
    df: pd.DataFrame,
    pending_indices: list[int],
    model: str,
    system_prompt: str,
    user_template: str,
    out_path: Path,
    temperature: float | None,
) -> None:
    """Judge every row in `pending_indices` concurrently, checkpointing periodically.

    Mutates `df` in place. Safe under asyncio's single-threaded cooperative
    model: each task only ever reads/writes its own row index, and the only
    suspension point is the `await call_judge(...)` call itself, so no two
    tasks' `df.at[...]` writes can interleave.

    Args:
        df: Full outputs DataFrame to judge and update in place.
        pending_indices: Row indices still needing a judgement.
        model: litellm model string for the judge.
        system_prompt: System prompt text.
        user_template: User prompt template to fill per row.
        out_path: CSV file path for periodic checkpoint saves.
        temperature: Sampling temperature for the judge calls, or None to
            omit it from the request and use the model's own default.
    """
    total = len(pending_indices)
    semaphore = asyncio.Semaphore(JUDGE_CONCURRENCY)
    completed = 0
    tasks = [
        _judge_one(position, idx, df, model, system_prompt, user_template, temperature, semaphore)
        for position, idx in enumerate(pending_indices, start=1)
    ]

    for task in asyncio.as_completed(tasks):
        position, idx, row, result = await task
        completed += 1

        if result is None:
            logger.warning("[%d/%d] idx=%d — judge call failed, skipping.", position, total, idx)
        else:
            df.at[idx, "eedi_score"] = result.overall_eedi_risk_score
            df.at[idx, "flagged_dimension"] = result.flagged_dimension
            df.at[idx, "eedi_justification"] = result.justification
            df.at[idx, "judge_raw_response"] = result.model_dump_json()

            logger.info(
                "[%d/%d] idx=%d | %s | %s | score=%s | flagged=%s",
                position,
                total,
                idx,
                row["platform"],
                row["condition_id"],
                result.overall_eedi_risk_score,
                result.flagged_dimension,
            )

        if completed % JUDGE_CHECKPOINT_EVERY == 0:
            df.dropna(subset=["eedi_score"]).to_csv(out_path, index=False)
            logger.info("Checkpoint saved (%d/%d).", completed, total)


def _redundancy_check(df: pd.DataFrame) -> None:
    """Log the correlation between EEDI scores and the normalised metric composite.

    Reports both Pearson r (assumes a linear relationship) and Spearman rho
    (only assumes a monotonic one) since eedi_score is an ordinal 1-5 scale,
    not a continuous variable -- Spearman is the more defensible statistic
    for it, with Pearson kept alongside since it's the more familiar one.

    Args:
        df: Judged outputs DataFrame with both eedi_score and metric columns.
    """
    # Deferred: scipy is not needed for `collect`/`judge` itself, only this check.
    logger.info("Loading scipy.stats module...")
    t0 = time.perf_counter()
    from scipy.stats import pearsonr, spearmanr

    logger.info("  module import took %.1fs", time.perf_counter() - t0)

    scored = df.dropna(subset=["eedi_score", "agency_ratio", "formality_score", "gain_loss_ratio", "semantic_distance"])
    if len(scored) < 10:
        logger.info("Too few scored rows for redundancy check.")
        return

    composite = (
        _minmax_scale(scored["agency_ratio"].abs())
        + _minmax_scale(scored["formality_score"])
        + _minmax_scale(scored["gain_loss_ratio"].abs())
        + _minmax_scale(scored["semantic_distance"])
    ) / 4

    scores = scored["eedi_score"].astype(float)
    r, p_pearson = pearsonr(scores, composite)
    rho, p_spearman = spearmanr(scores, composite)

    # Spearman is the primary read (ordinal-appropriate); Pearson kept for
    # familiarity. Same 0.9 threshold applied to whichever is higher, since
    # either one being that high would indicate mechanical aggregation.
    verdict = (
        "Judge appears to add integrated judgement beyond the composite."
        if max(r, rho) < 0.9
        else "High correlation — judge may be mechanically aggregating metrics. Review prompting."
    )
    logger.info(
        "Redundancy check — Spearman rho(EEDI score, metric composite) = %.3f (p=%.4f), "
        "Pearson r = %.3f (p=%.4f), n=%d. %s",
        rho,
        p_spearman,
        r,
        p_pearson,
        len(scored),
        verdict,
    )


def add_parser(subparsers) -> None:
    """Register the `judge` subcommand and its arguments.

    Args:
        subparsers: Subparser action from the parent ArgumentParser.
    """
    p = subparsers.add_parser("judge", help="Run LLM judge to assign EEDI risk scores.")
    p.add_argument("--in", dest="input", default="data/scored_outputs_with_breed.csv")
    p.add_argument("--out", dest="output", default="data/judged_outputs.csv")
    p.add_argument(
        "--model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"litellm model string for the judge (default: {DEFAULT_JUDGE_MODEL}).",
    )
    p.add_argument("--resume", action="store_true", help="Skip rows that already have an eedi_score in --out.")
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature for the judge. Omitted by default, so reasoning-family "
        "models (e.g. gpt-5.x) use their own default rather than needing an explicit "
        "value. Pass e.g. --temperature 0 for deterministic scoring on a non-reasoning model.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only judge the first N pending rows -- for a quick test run before "
        "committing to a full pass. Omitted by default (no limit).",
    )


def run(args: argparse.Namespace) -> None:
    """Run the LLM judge over all unscored rows and write the judged CSV.

    Args:
        args: Parsed CLI arguments from add_parser.
    """
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    logger.info("Loaded %d rows from %s", len(df), in_path)

    _validate_columns(df)
    logger.info("Input columns validated.")

    if args.resume and out_path.exists():
        prev = pd.read_csv(out_path)
        logger.info("Resuming — %d rows already judged in %s.", len(prev), out_path)
        df = df.merge(prev[_KEY_COLS + _JUDGE_COLS], on=_KEY_COLS, how="left")
    else:
        for col in _JUDGE_COLS:
            df[col] = pd.NA

    logger.info("Loading judge prompts...")
    system_prompt = load_prompt("eedi_judge_system", subdir="agents")
    user_template = load_prompt("eedi_judge_user", subdir="agents")

    pending_indices = df[df["eedi_score"].isna()].index.tolist()

    if args.limit is not None:
        pending_indices = pending_indices[: args.limit]
        logger.info("--limit set: only judging the first %d pending rows.", args.limit)

    logger.info(
        "%d rows to judge with %s (concurrency=%d, temperature=%s)",
        len(pending_indices),
        args.model,
        JUDGE_CONCURRENCY,
        args.temperature,
    )

    asyncio.run(_run_judging(df, pending_indices, args.model, system_prompt, user_template, out_path, args.temperature))

    successful = df.dropna(subset=["eedi_score"])
    successful.to_csv(out_path, index=False)
    logger.info(
        "Saved %d/%d rows to %s (%d failed the judge call and were left out — rerun with " "--resume to retry them).",
        len(successful),
        len(df),
        out_path,
        len(df) - len(successful),
    )

    _redundancy_check(successful)
