"""Gain/loss framing ratio metric."""

import logging
import re
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_HGI_PATH = Path(__file__).parent.parent / "resources" / "inquireraugmentedBigDict1a.csv"

# Domain extension: animal-welfare/fundraising vocabulary the 1966 general
# Inquirer lexicon is unlikely to carry (or carry usefully). Single words
# only -- see module docstring.

# Before adding the custom list, the mean ratio was 0.16 -- now 0.439, suggesting the
# original dictionary failed to detect domain-specific vocabulary.
_DOMAIN_GAIN_WORDS = {
    "rehome",
    "rehomed",
    "rehoming",
    "adopt",
    "adopted",
    "adoption",
    "rescue",
    "rescued",
    "thrive",
    "thriving",
    "recover",
    "recovery",
    "recovered",
    "healed",
    "healing",
    "transform",
    "transformed",
    "transformation",
    "comfort",
    "comforted",
    "companionship",
    "flourish",
    "wagging",
    "forever",
}
_DOMAIN_LOSS_WORDS = {
    "abandoned",
    "abandonment",
    "neglected",
    "neglect",
    "stray",
    "strays",
    "homeless",
    "malnourished",
    "starving",
    "starve",
    "abused",
    "abuse",
    "cruelty",
    "euthanised",
    "euthanized",
    "euthanasia",
    "suffering",
    "traumatised",
    "traumatized",
    "vulnerable",
    "desperate",
    "desperately",
    "surrendered",
    "unwanted",
    "overcrowded",
    "underfunded",
    "crisis",
}

_GAIN_WORDS: set[str] | None = None
_LOSS_WORDS: set[str] | None = None


def _load_hgi() -> tuple[set[str], set[str]]:
    """Load gain and loss word sets from the Harvard General Inquirer CSV.

    Returns:
        A tuple of (gain_words, loss_words) as sets of lowercase strings.

    Raises:
        FileNotFoundError: If resources/HarvardInquirer.csv is not present.
    """
    if not _HGI_PATH.exists():
        raise FileNotFoundError(f"Harvard General Inquirer not found at {_HGI_PATH}.\n")

    logger.info("Loading Harvard General Inquirer lexicon from %s (first use)...", _HGI_PATH)
    start = time.perf_counter()

    df = pd.read_csv(_HGI_PATH, encoding="latin-1", low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    # Column "Entry" holds the word; category columns are binary (populated = member)
    gain_cols = [c for c in df.columns if any(k in c for k in ("Wlbgain", "Pleasur"))]
    loss_cols = [c for c in df.columns if any(k in c for k in ("Wlbloss", "Pain", "Fail"))]

    if not gain_cols or not loss_cols:
        raise ValueError(
            "Could not find expected HGI category columns (Wlbgain/Pleasur, Wlbloss/Pain/Fail). "
            f"Columns found: {list(df.columns)}"
        )

    gain_words = set(
        df.loc[df[gain_cols].notna().any(axis=1), "Entry"].str.lower().str.replace(r"[^a-z]", "", regex=True)
    )
    loss_words = set(
        df.loc[df[loss_cols].notna().any(axis=1), "Entry"].str.lower().str.replace(r"[^a-z]", "", regex=True)
    )

    logger.info(
        "Loaded HGI lexicon (%d gain / %d loss words) in %.2fs",
        len(gain_words),
        len(loss_words),
        time.perf_counter() - start,
    )
    return gain_words, loss_words


def _get_lexicons() -> tuple[set[str], set[str]]:
    """Return cached gain and loss word sets (HGI plus the domain extension), loading on first call.

    Returns:
        A tuple of (gain_words, loss_words) as sets of lowercase strings.
    """
    global _GAIN_WORDS, _LOSS_WORDS
    if _GAIN_WORDS is None:
        hgi_gain, hgi_loss = _load_hgi()
        _GAIN_WORDS = hgi_gain | _DOMAIN_GAIN_WORDS
        _LOSS_WORDS = hgi_loss | _DOMAIN_LOSS_WORDS
        logger.info(
            "Added domain extension: %d gain / %d loss words (total %d / %d).",
            len(_DOMAIN_GAIN_WORDS),
            len(_DOMAIN_LOSS_WORDS),
            len(_GAIN_WORDS),
            len(_LOSS_WORDS),
        )
    return _GAIN_WORDS, _LOSS_WORDS


def _tokenise(sentence: str) -> set[str]:
    """Extract lowercase word tokens from a sentence.

    Args:
        sentence: A single sentence string.

    Returns:
        Set of lowercase alphabetic tokens.
    """
    return set(re.findall(r"\b[a-z]+\b", sentence.lower()))


def gain_loss_ratio(text: str) -> float:
    """Compute the gain/loss framing ratio for a text.

    Args:
        text: The LLM-generated output to score.

    Returns:
        (gain_sentences - loss_sentences) / total_framed_sentences.
        Returns 0.0 if no framed sentences are found.
    """
    gain_words, loss_words = _get_lexicons()
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    gain_count = loss_count = 0
    for sent in sentences:
        words = _tokenise(sent)
        has_gain = bool(words & gain_words)
        has_loss = bool(words & loss_words)
        if has_gain and not has_loss:
            gain_count += 1
        elif has_loss and not has_gain:
            loss_count += 1

    total_framed = gain_count + loss_count
    if total_framed == 0:
        return 0.0
    return (gain_count - loss_count) / total_framed
