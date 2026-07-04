"""Structured logging setup for VIOLA-SO101.

One configured root logger: console + rotating file handler. Only the main
process logs (silence everything else in distributed runs). Python warnings are
routed through logging. Level is configurable via argument or the
``VIOLA_LOG_LEVEL`` environment variable.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path

_CONFIGURED = False


def setup_logging(
    log_dir: str | Path | None = None,
    level: str | int | None = None,
    is_main_process: bool = True,
    filename: str = "viola.log",
) -> logging.Logger:
    """Configure the root logger. Safe to call more than once (idempotent).

    On non-main processes, logging is raised to ERROR so worker ranks stay quiet.
    """
    global _CONFIGURED
    root = logging.getLogger()

    if level is None:
        level = os.environ.get("VIOLA_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    if not is_main_process:
        root.setLevel(logging.ERROR)
        return logging.getLogger("viola_so101")

    if _CONFIGURED:
        root.setLevel(level)
        return logging.getLogger("viola_so101")

    root.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fileh = logging.handlers.RotatingFileHandler(
            log_dir / filename, maxBytes=10 * 1024 * 1024, backupCount=5,
        )
        fileh.setFormatter(fmt)
        root.addHandler(fileh)

    # Route warnings.warn(...) through logging instead of stderr.
    logging.captureWarnings(True)
    _CONFIGURED = True
    return logging.getLogger("viola_so101")
