"""Centralized logging configuration without leaking user data."""

import logging
import sys

from ml_app.config import settings


def configure_logging() -> logging.Logger:
    """Configure and return the root application logger."""
    logger = logging.getLogger("ml_app")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)
    return logger


logger = configure_logging()
