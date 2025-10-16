"""Logging utilities built on top of :mod:`loguru`."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logging(log_file: Optional[Path] = None, level: str = "INFO") -> None:
    """Configure the global loguru logger.

    Args:
        log_file: Optional file path for log sink.
        level: Minimum log level (string understood by loguru).
    """

    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=level)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, level=level, rotation="10 MB", retention="7 days")


__all__ = ["setup_logging", "logger"]
