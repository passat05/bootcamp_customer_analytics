"""
logging_utils.py – Shared logging setup for scripts and the API service.

Configures console + per-run file logging so runs can be inspected after
the fact instead of only living in stdout.
"""

import logging
import os
from datetime import datetime

LOG_DIR = os.environ.get("LOG_DIR", "logs")


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure root logging with a console handler and a timestamped file handler.

    Log files are written to LOG_DIR/{name}_{YYYYmmdd_HHMMSS}.log so each run
    is kept as a separate, retrievable record.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"{name}_{timestamp}.log")

    formatter = logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = [console_handler, file_handler]

    logger = logging.getLogger(name)
    logger.info("Logging to %s", log_file)
    return logger
