"""Formality score metric."""

import logging
import time

import spacy

logger = logging.getLogger(__name__)

_nlp: spacy.language.Language | None = None

_FORMAL_POS = {"NOUN", "PROPN", "ADJ", "ADP", "DET"}
_INFORMAL_POS = {"PRON", "VERB", "AUX", "ADV", "INTJ"}


def _get_nlp() -> spacy.language.Language:
    """Load and cache the spaCy pipeline on first call.

    Returns:
        A spaCy Language object with NER and parser disabled for speed.
    """
    global _nlp
    if _nlp is None:
        logger.info("Loading spaCy model 'en_core_web_sm' (first use)...")
        start = time.perf_counter()
        _nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        logger.info("Loaded spaCy model in %.1fs", time.perf_counter() - start)
    return _nlp


def _score_doc(doc: spacy.tokens.Doc) -> float:
    """Compute the formality F-score from an already-tagged spaCy Doc.

    Args:
        doc: A spaCy Doc for one text.

    Returns:
        F-score in the range [0, 100]. Returns 50.0 for empty input.
    """
    tokens = [t for t in doc if not t.is_space and not t.is_punct]
    if not tokens:
        return 50.0

    n = len(tokens)
    formal_count = sum(1 for t in tokens if t.pos_ in _FORMAL_POS)
    informal_count = sum(1 for t in tokens if t.pos_ in _INFORMAL_POS)

    formal_freq = formal_count / n * 100
    informal_freq = informal_count / n * 100

    return (formal_freq - informal_freq + 100) / 2


def formality_score(text: str) -> float:
    """Compute the Heylighen & Deacon formality F-score for a single text.

    Args:
        text: The LLM-generated output to score.

    Returns:
        F-score in the range [0, 100]. Returns 50.0 for empty input.
    """
    return _score_doc(_get_nlp()(text))


def formality_scores(texts: list[str]) -> list[float]:
    """Compute formality F-scores for many texts in one batched spaCy pass.

    Faster than calling formality_score() once per text — nlp.pipe() batches
    tokenization/tagging across all texts instead of paying spaCy's per-call
    overhead for each one individually.

    Args:
        texts: LLM-generated outputs to score.

    Returns:
        F-scores in the range [0, 100], one per entry in texts (same order).
    """
    return [_score_doc(doc) for doc in _get_nlp().pipe(texts)]
