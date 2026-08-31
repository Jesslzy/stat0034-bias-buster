"""Semantic distance metric."""

import logging
import time

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_MODEL_NAME = "all-mpnet-base-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Load and cache the sentence transformer model on first call.

    Returns:
        A SentenceTransformer instance using all-mpnet-base-v2.
    """
    global _model
    if _model is None:
        logger.info(
            "Loading sentence-transformers model '%s' (first use; " "downloads automatically if not already cached)...",
            _MODEL_NAME,
        )
        start = time.perf_counter()
        _model = SentenceTransformer(_MODEL_NAME)
        logger.info("Loaded sentence-transformers model in %.1fs", time.perf_counter() - start)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Encode a list of texts into normalised sentence embeddings.

    Args:
        texts: List of strings to encode.

    Returns:
        Array of shape (n, d) with L2-normalised embeddings.
    """
    model = _get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(embeddings)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two normalised vectors.

    Args:
        a: First normalised embedding vector.
        b: Second normalised embedding vector.

    Returns:
        Cosine distance in [0, 2]. A value of 0 means identical, 1 means
        orthogonal, and 2 means opposite directions.
    """
    similarity = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return 1.0 - similarity


def compute_semantic_distances(
    texts: list[str],
    reference_text: str,
) -> list[float]:
    """Compute cosine distances of a list of texts from a reference text.

    All texts and the reference are encoded in a single batch for efficiency.

    Args:
        texts: List of LLM-generated outputs to score.
        reference_text: The reference condition output to measure distance from.

    Returns:
        List of cosine distances, one per entry in texts.
    """
    all_texts = [reference_text] + texts
    embeddings = embed(all_texts)
    ref_emb = embeddings[0]
    return [cosine_distance(ref_emb, emb) for emb in embeddings[1:]]
