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
    formatter = logging.Formatter(log_format)

    root = logging.getLogger()
    root.setLevel(log_level)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    if verbose:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        root.addHandler(sh)


