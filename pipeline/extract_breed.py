"""Extract structured dog characteristics from generated letters."""

import argparse
import asyncio
import logging
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from constants import BREED_CHECKPOINT_EVERY, BREED_CONCURRENCY, DEFAULT_BREED_MODEL
from models.models import BreedExtraction
from utils.llm_client import acall_with_retry
from utils.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

_KEY_COLS = ["condition_id", "platform", "replicate"]
_BREED_COLS = [
    "dog_name",
    "breed_mentioned",
    "kc_breed_group",
    "size_category",
    "life_stage",
    "breed_raw_response",
]


def _validate_columns(df: pd.DataFrame) -> None:
    """Raise if any required columns are missing from the input DataFrame.

    Args:
        df: Input DataFrame to validate.

    Raises:
        ValueError: If any required columns are absent.
    """
    required = {"response_text", "platform", "condition_id", "replicate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input csv is missing columns: {missing}.")


async def call_extractor(model: str, system: str, user: str) -> BreedExtraction | None:
    """Call the extraction model with tenacity-managed exponential back-off retry.

    Args:
        model: litellm model string for the extractor.
        system: System prompt text.
        user: User prompt text for this row.

    Returns:
        Parsed BreedExtraction, or None if all attempts fail, an auth error
        occurs, or the response doesn't validate against the schema.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    raw = await acall_with_retry(model, messages, temperature=0.0, max_tokens=500, response_format=BreedExtraction)
    if raw is None:
        return None
    try:
        return BreedExtraction.model_validate_json(raw)
    except ValidationError as exc:
        logger.warning("Breed extraction failed schema validation: %s\n%s", exc, raw)
        return None


async def extract_breed_from_text(
    response_text: str,
    model: str = DEFAULT_BREED_MODEL,
    system_prompt: str | None = None,
    user_template: str | None = None,
) -> BreedExtraction | None:
    """Extract structured dog details from one fundraising text.

    This is the shared single-text entry point used by both the batch pipeline
    and the Streamlit app. Callers processing many rows should load the prompts
    once and pass them in to avoid repeated file reads.

    Args:
        response_text: AI-generated fundraising text to examine.
        model: LiteLLM model identifier used for extraction.
        system_prompt: Optional pre-loaded extraction system prompt.
        user_template: Optional pre-loaded extraction user template.

    Returns:
        Validated dog details, or None when extraction fails after retries.

    Raises:
        ValueError: If response_text is empty.
    """
    response_text = response_text.strip()
    if not response_text:
        raise ValueError("response_text must not be empty.")

    system_prompt = system_prompt or load_prompt("breed_extraction_system", subdir="agents")
    user_template = user_template or load_prompt("breed_extraction_user", subdir="agents")
    user_prompt = user_template.format(response_text=response_text)
    return await call_extractor(model, system_prompt, user_prompt)


async def _extract_one(
    position: int,
    idx: int,
    df: pd.DataFrame,
    model: str,
    system_prompt: str,
    user_template: str,
    semaphore: asyncio.Semaphore,
) -> tuple[int, int, pd.Series, BreedExtraction | None]:
    """Extract dog details for one row without mutating shared state.

    Args:
        position: Position of the row in the pending work.
        idx: DataFrame index of the row to process.
        df: Source data containing the generated letter.
        model: LiteLLM model identifier used for extraction.
        system_prompt: Extraction system prompt.
        user_template: Extraction user prompt template.
        semaphore: Limit on concurrent extraction calls.

    Returns:
        The work position, row index, source row, and extraction result.
    """
    row = df.loc[idx]
    async with semaphore:
        result = await extract_breed_from_text(
            response_text=row["response_text"],
            model=model,
            system_prompt=system_prompt,
            user_template=user_template,
        )
    return position, idx, row, result


async def _run_extraction(
    df: pd.DataFrame,
    pending_indices: list[int],
    model: str,
    system_prompt: str,
    user_template: str,
    out_path: Path,
) -> None:
    """Extract breed info for every row in `pending_indices`, checkpointing periodically.

    Mutates `df` in place. Safe under asyncio's single-threaded cooperative
    model: each task only ever reads/writes its own row index, and the only
    suspension point is the `await call_extractor(...)` call itself, so no
    two tasks' `df.at[...]` writes can interleave.

    Args:
        df: Full outputs DataFrame to extract into and update in place.
        pending_indices: Row indices still needing extraction.
        model: litellm model string for the extractor.
        system_prompt: System prompt text.
        user_template: User prompt template to fill per row.
        out_path: CSV file path for periodic checkpoint saves.
    """
    total = len(pending_indices)
    semaphore = asyncio.Semaphore(BREED_CONCURRENCY)
    completed = 0
    tasks = [
        _extract_one(position, idx, df, model, system_prompt, user_template, semaphore)
        for position, idx in enumerate(pending_indices, start=1)
    ]

    for task in asyncio.as_completed(tasks):
        position, idx, row, result = await task
        completed += 1

        if result is None:
            logger.warning("[%d/%d] idx=%d — extraction call failed, skipping.", position, total, idx)
        else:
            df.at[idx, "dog_name"] = result.dog_name
            df.at[idx, "breed_mentioned"] = result.breed_mentioned
            df.at[idx, "kc_breed_group"] = result.kc_breed_group
            df.at[idx, "size_category"] = result.size_category
            df.at[idx, "life_stage"] = result.life_stage
            df.at[idx, "breed_raw_response"] = result.model_dump_json()

            logger.info(
                "[%d/%d] idx=%d | %s | %s | group=%s | size=%s | stage=%s",
                position,
                total,
                idx,
                row["platform"],
                row["condition_id"],
                result.kc_breed_group,
                result.size_category,
                result.life_stage,
            )

        if completed % BREED_CHECKPOINT_EVERY == 0:
            df.dropna(subset=["kc_breed_group"]).to_csv(out_path, index=False)
            logger.info("Checkpoint saved (%d/%d).", completed, total)


def add_parser(subparsers) -> None:
    """Register the `extract-breed` subcommand and its arguments.

    Args:
        subparsers: Subparser action from the parent ArgumentParser.
    """
    p = subparsers.add_parser("extract-breed", help="Extract dog breed/size/life-stage info from generated letters.")
    p.add_argument("--in", dest="input", default="data/scored_outputs.csv")
    p.add_argument("--out", dest="output", default="data/scored_outputs_with_breed.csv")
    p.add_argument(
        "--model",
        default=DEFAULT_BREED_MODEL,
        help=f"litellm model string for extraction (default: {DEFAULT_BREED_MODEL}).",
    )
    p.add_argument("--resume", action="store_true", help="Skip rows that already have a kc_breed_group in --out.")


def run(args: argparse.Namespace) -> None:
    """Run breed extraction over all pending rows and write the enriched CSV.

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
        logger.info("Resuming — %d rows already extracted in %s.", len(prev), out_path)
        df = df.merge(prev[_KEY_COLS + _BREED_COLS], on=_KEY_COLS, how="left")
    else:
        for col in _BREED_COLS:
            df[col] = pd.NA

    logger.info("Loading extraction prompts...")
    system_prompt = load_prompt("breed_extraction_system", subdir="agents")
    user_template = load_prompt("breed_extraction_user", subdir="agents")

    pending_indices = df[df["kc_breed_group"].isna()].index.tolist()
    logger.info("%d rows to extract with %s (concurrency=%d)", len(pending_indices), args.model, BREED_CONCURRENCY)

    asyncio.run(_run_extraction(df, pending_indices, args.model, system_prompt, user_template, out_path))

    successful = df.dropna(subset=["kc_breed_group"])
    successful.to_csv(out_path, index=False)
    logger.info(
        "Saved %d/%d rows to %s (%d failed the extraction call and were left out — rerun with "
        "--resume to retry them).",
        len(successful),
        len(df),
        out_path,
        len(df) - len(successful),
    )
