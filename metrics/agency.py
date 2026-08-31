"""Agency ratio metric."""

import logging
import re
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_LEXICON_PATH = Path(__file__).parent.parent / "resources" / "agency_power.csv"

_LEXICON: dict[str, str] | None = None


def _load_lexicon() -> dict[str, str]:
    """Load the Connotation Frames lexicon from disk.

    Returns:
        Mapping of lowercase verb to "high_agency" or "low_agency".

    Raises:
        FileNotFoundError: If resources/agency_power.csv is not present.
    """
    if not _LEXICON_PATH.exists():
        raise FileNotFoundError(
            f"Agency lexicon not found at {_LEXICON_PATH}.\n"
            "Download agency_power.csv from maartensap.com/connotation-frames "
            "and save it as resources/agency_power.csv."
        )

    logger.info("Loading agency lexicon from %s (first use)...", _LEXICON_PATH)
    start = time.perf_counter()

    df = pd.read_csv(_LEXICON_PATH)
    lexicon: dict[str, str] = {}
    for _, row in df.iterrows():
        verb = str(row["verb"]).lower().strip()
        agency = str(row.get("agency", "")).strip()
        if agency == "agency_pos":
            lexicon[verb] = "high_agency"
        elif agency == "agency_neg":
            lexicon[verb] = "low_agency"

    logger.info("Loaded agency lexicon (%d terms) in %.2fs", len(lexicon), time.perf_counter() - start)
    return lexicon


def _get_lexicon() -> dict[str, str]:
    """Return the cached lexicon, loading it on first call.

    Returns:
        Mapping of lowercase verb to "high_agency" or "low_agency".
    """
    global _LEXICON
    if _LEXICON is None:
        _LEXICON = _load_lexicon()
    return _LEXICON


def agency_ratio(text: str) -> float:
    """Compute the agency ratio for a text.

    Args:
        text: The LLM-generated output to score.

    Returns:
        (high_agency_count - low_agency_count) / total_coded_count.
        Returns 0.0 if no coded terms are found.
    """
    lexicon = _get_lexicon()
    tokens = re.findall(r"\b[a-z]+\b", text.lower())
    high_agency = sum(1 for t in tokens if lexicon.get(t) == "high_agency")
    low_agency = sum(1 for t in tokens if lexicon.get(t) == "low_agency")
    total = high_agency + low_agency
    if total == 0:
        return 0.0
    return (high_agency - low_agency) / total
