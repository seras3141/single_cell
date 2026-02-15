import logging
import sys, os
from typing import Dict, Any
from pathlib import Path

def setup_logging(level: str = "INFO", log_file = None, verbose: bool = True, log_config: Dict[str, Any] = {}) -> None:
    """Setup logging configuration."""

    log_level = level or log_config.get('level', 'INFO')
    if isinstance(log_level, str):
        log_level = log_level.upper()

    log_file = log_file or log_config.get('log_file', None)
    
    if log_file:
        log_dir = Path(log_file).parent
        os.makedirs(log_dir, exist_ok=True)

    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Set up handlers
    handlers = []
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    if verbose:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )


