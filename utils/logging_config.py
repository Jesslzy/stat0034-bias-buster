"""Configure logging for the pipeline."""

import logging


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a standard format.

    Args:
        level: Logging level (e.g. logging.DEBUG, logging.INFO). Defaults to INFO.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
