import logging
import sys, os
from typing import Dict, Any
from pathlib import Path

def setup_logging(level: str = "INFO", log_file = None, log_config: Dict[str, Any] = {}) -> None:
    """Setup logging configuration."""

    log_level = level or log_config.get('level', 'INFO')
    if isinstance(log_level, str):
        log_level = log_level.upper()

    log_file = log_file or log_config.get('log_file', None)

    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(
            level=log_level,
            format=log_format
        )


