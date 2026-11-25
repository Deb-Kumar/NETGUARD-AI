"""Centralized logging configuration for NETGUARD AI."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

from src.utils.config import ProjectPaths, ensure_directories, get_paths


def configure_logging(
    name: str = "netguard",
    *,
    level: str | None = None,
    log_to_file: bool = True,
    paths: ProjectPaths | None = None,
) -> logging.Logger:
    """Configure and return a centralized logger writing to console and logs/netguard.log."""
    resolved_level_name = (level or os.getenv("NETGUARD_LOG_LEVEL", "INFO")).upper()
    numeric_level = getattr(logging, resolved_level_name, logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(numeric_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_to_file:
        resolved_paths = ensure_directories(paths or get_paths())
        log_file = resolved_paths.logs_dir / "netguard.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=2 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
