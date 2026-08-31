"""Collect generated letters across experimental conditions."""

import argparse
import asyncio
import itertools
import logging
import re
from pathlib import Path

import pandas as pd

from constants import AGE_GROUPS, CONCURRENCY_PER_PLATFORM, GENDERS, PLATFORMS, TENURE_PROFILES
from models.models import PromptCondition
from utils.llm_client import acall_with_retry
from utils.prompt_loader import load_prompt, save_prompt

logger = logging.getLogger(__name__)


def _build_conditions(stage: str) -> list[PromptCondition]:
    """Build the full list of prompt conditions for the requested stage(s).

    Args:
        stage: One of "1", "2", or "all".

    Returns:
        List of PromptCondition objects covering all variable combinations.
    """
    conditions = []
    if stage in ("1", "all"):
        for gender, age in itertools.product(GENDERS, AGE_GROUPS):
            conditions.append(PromptCondition(stage="stage1", gender=gender, age_group=age))
    if stage in ("2", "all"):
        for gender, age, tenure in itertools.product(GENDERS, AGE_GROUPS, TENURE_PROFILES):
            conditions.append(PromptCondition(stage="stage2", gender=gender, age_group=age, tenure=tenure))
    return conditions


def _condition_filename(condition: PromptCondition) -> str:
    """Build a filesystem-safe filename stem identifying one condition.

    Args:
        condition: The experimental condition to name.

    Returns:
        A lowercase, hyphen/underscore-safe filename stem (no extension).
    """
    parts = [condition.stage, condition.gender, condition.age_group]
    if condition.tenure:
        parts.append(condition.tenure)
    return "_".join(re.sub(r"[^a-z0-9]+", "-", part.lower()).strip("-") for part in parts)


def _build_prompt(condition: PromptCondition, stage1_template: str | None, stage2_template: str | None) -> str:
    """Fill the appropriate template with the condition's variable values.

    Args:
        condition: The experimental condition supplying the variable values.
        stage1_template: Raw template string for Stage 1 prompts, or None if
            Stage 1 wasn't requested.
        stage2_template: Raw template string for Stage 2 prompts, or None if
            Stage 2 wasn't requested.

    Returns:
        The fully filled prompt string ready to send to the model.
    """
    if condition.stage == "stage1":
        return stage1_template.format(GENDER=condition.gender, AGE=condition.age_group)
    return stage2_template.format(GENDER=condition.gender, AGE=condition.age_group, TENURE=condition.tenure)


async def _query_with_retry(model_id: str, prompt: str) -> str | None:
    """Query a model with tenacity-managed exponential back-off retry.

    Args:
        model_id: litellm model string (e.g. "openai/gpt-4o").
        prompt: User prompt text to send.

    Returns:
        The model's response text, or None if all attempts fail or auth error occurs.
    """
    return await acall_with_retry(model_id, [{"role": "user", "content": prompt}], temperature=1.0)


def _iter_work_items(conditions: list[PromptCondition], platforms: list[str], replicates: int):
    """Enumerate every (condition, platform, replicate) combination to run.

    Args:
        conditions: All prompt conditions for this run.
        platforms: Short platform names to query.
        replicates: Number of replicates per condition/platform.

    Yields:
        (condition, platform_name, replicate_number) tuples, 1-indexed on replicate.
    """
    yield from itertools.product(conditions, platforms, range(1, replicates + 1))


async def _collect_row(
    condition: PromptCondition, platform_name: str, model_id: str, rep: int, prompt_text: str
) -> dict | None:
    """Query one platform for one condition/replicate and build the result row.

    Args:
        condition: The experimental condition being queried.
        platform_name: Short platform name (e.g. "gpt").
        model_id: litellm model string for this platform.
        rep: Replicate number (1-indexed).
        prompt_text: The prompt to send.

    Returns:
        A result row dict, or None if the query failed.
    """
    response_text = await _query_with_retry(model_id, prompt_text)
    if response_text is None:
        logger.warning("Skipping failed: %s", (condition.condition_id, platform_name, rep))
        return None

    return {
        "stage": condition.stage,
        "gender": condition.gender,
        "age_group": condition.age_group,
        "tenure": condition.tenure,
        "condition_id": condition.condition_id,
        "platform": platform_name,
        "model_id": model_id,
        "replicate": rep,
        "prompt_text": prompt_text,
        "response_text": response_text,
    }


def _append_row(row: dict, out_path: Path) -> None:
    """Append one collected row to the output CSV, writing a header if the file is new.

    Safe to call from concurrent asyncio tasks: this function contains no
    `await`, so a single call always runs to completion before any other
    task's code can run on the (single-threaded) event loop.

    Args:
        row: The result row to append.
        out_path: CSV file path to append to.
    """
    write_header = not out_path.exists()
    pd.DataFrame([row]).to_csv(out_path, mode="a", header=write_header, index=False)


async def _collect_work_item(
    completed: int,
    condition: PromptCondition,
    platform_name: str,
    rep: int,
    existing: set[tuple],
    prompt: str,
    semaphore: asyncio.Semaphore,
) -> tuple[int, PromptCondition, str, int, dict | None] | None:
    """Collect one experimental output without updating shared counters.

    Args:
        completed: Position of the item in the complete work list.
        condition: Experimental condition for the output.
        platform_name: Short name of the generative AI platform.
        rep: Replicate number.
        existing: Keys for outputs that have already been collected.
        prompt: Fully rendered prompt for the condition.
        semaphore: Concurrency limit for the platform.

    Returns:
        Work metadata and the collected row, or ``None`` for an existing row.
    """
    if (condition.condition_id, platform_name, rep) in existing:
        return None

    async with semaphore:
        row = await _collect_row(condition, platform_name, PLATFORMS[platform_name], rep, prompt)
    return completed, condition, platform_name, rep, row


async def _run_collection(
    work_items: list[tuple[PromptCondition, str, int]],
    existing: set[tuple],
    prompts: dict[str, str],
    out_path: Path,
    platforms: list[str],
) -> int:
    """Run every work item concurrently, bounded per platform, appending rows as they finish.

    Args:
        work_items: All (condition, platform_name, replicate) tuples to process.
        existing: Set of (condition_id, platform_name, replicate) already collected.
        prompts: Mapping of condition_id to its fully built prompt text.
        out_path: CSV file path to append completed rows to.
        platforms: Short platform names being queried, for sizing the semaphores.

    Returns:
        Number of new rows collected.
    """
    total = len(work_items)
    semaphores = {p: asyncio.Semaphore(CONCURRENCY_PER_PLATFORM) for p in platforms}
    new_count = 0
    tasks = [
        _collect_work_item(
            completed,
            condition,
            platform_name,
            rep,
            existing,
            prompts[condition.condition_id],
            semaphores[platform_name],
        )
        for completed, (condition, platform_name, rep) in enumerate(work_items, start=1)
    ]

    for task in asyncio.as_completed(tasks):
        result = await task
        if result is None:
            continue
        completed, condition, platform_name, rep, row = result
        if row is None:
            continue
        _append_row(row, out_path)
        new_count += 1
        logger.info("[%d/%d] %s | %s | rep %d", completed, total, condition.condition_id, platform_name, rep)
    return new_count


def add_parser(subparsers) -> None:
    """Register the `collect` subcommand and its arguments.

    Args:
        subparsers: Subparser action from the parent ArgumentParser.
    """
    p = subparsers.add_parser("collect", help="Collect LLM outputs for all prompt conditions.")
    p.add_argument("--stage", choices=["1", "2", "all"], default="all", help="Which stage(s) to run (default: all).")
    p.add_argument(
        "--replicates",
        type=int,
        default=5,
        help="Number of times to query each condition per platform (default: 5).",
    )
    p.add_argument("--out", default="data/raw_outputs.csv", help="Output path for collected data.")
    p.add_argument(
        "--platforms",
        nargs="+",
        choices=list(PLATFORMS),
        default=list(PLATFORMS),
        help="Which platforms to query.",
    )
    p.add_argument("--resume", action="store_true", help="Skip rows already present in --out (resume interrupted run).")


def run(args: argparse.Namespace) -> None:
    """Execute the data collection loop and write results to CSV.

    Args:
        args: Parsed CLI arguments from add_parser.
    """
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    conditions = _build_conditions(args.stage)
    logger.info(
        "Total conditions: %d | stage: %s | platforms: %s | replicates: %d",
        len(conditions),
        args.stage,
        args.platforms,
        args.replicates,
    )

    existing: set[tuple] = set()
    if args.resume and out_path.exists():
        df_existing = pd.read_csv(out_path)
        existing = set(zip(df_existing["condition_id"], df_existing["platform"], df_existing["replicate"]))
        logger.info("Resuming — %d rows already collected.", len(df_existing))
    elif out_path.exists():
        # Fresh (non-resumed) run starts the output file over, matching the
        # old end-of-run overwrite behaviour.
        out_path.unlink()

    stage1_template = load_prompt("stage1") if args.stage in ("1", "all") else None
    stage2_template = load_prompt("stage2") if args.stage in ("2", "all") else None
    prompts = {}
    for condition in conditions:
        prompt_text = _build_prompt(condition, stage1_template, stage2_template)
        prompts[condition.condition_id] = prompt_text
        save_prompt(_condition_filename(condition), prompt_text)

    work_items = list(_iter_work_items(conditions, args.platforms, args.replicates))
    new_count = asyncio.run(_run_collection(work_items, existing, prompts, out_path, args.platforms))

    if new_count == 0:
        logger.info("No new rows collected.")
    else:
        logger.info("Saved %d new rows to %s", new_count, out_path)
