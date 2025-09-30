"""Logging utilities for DF-HGNN."""
from __future__ import annotations

import logging
import os
from typing import Optional


def setup_logging(level: str = "INFO", fmt: Optional[str] = None) -> None:
    """Configure global logging format and level.

    Args:
        level: Logging level name.
        fmt: Optional logging format string.
    """
    if fmt is None:
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=fmt)
    os.environ.setdefault("DF_HGNN_LOG_LEVEL", level.upper())


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger."""
    return logging.getLogger(name)
