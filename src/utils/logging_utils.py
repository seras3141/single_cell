import logging
import sys, os
from typing import Dict, Any, Optional
from pathlib import Path

_LOGGING_CONFIGURED = False

_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


def setup_logging(
    level: str = "INFO",
    log_file=None,
    verbose: bool = True,
    log_config: Dict[str, Any] = {},
    force: bool = False,
) -> None:
    """Configure the root logger. Call this ONCE, at the program entry point.

    Library/submodule code must NOT call this; it should only do
    ``logging.getLogger(__name__)`` and rely on propagation to the root logger.

    Idempotent by design: repeated calls are no-ops unless ``force=True``. This
    guards against accidental double-configuration (Jupyter re-imports,
    multiprocessing workers re-importing modules, tests calling ``main()`` twice)
    which would otherwise stack duplicate handlers and emit every record twice.
    """
    global _LOGGING_CONFIGURED

    if _LOGGING_CONFIGURED and not force:
        return

    log_level = level or log_config.get('level', 'INFO')
    if isinstance(log_level, str):
        log_level = log_level.upper()

    log_file = log_file or log_config.get('log_file', None)

    if log_file:
        log_dir = Path(log_file).parent
        os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(_LOG_FORMAT)

    root = logging.getLogger()

    # Remove any handlers left over from a previous configuration so we never
    # stack duplicates (belt-and-suspenders alongside the guard above).
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()

    root.setLevel(log_level)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    if verbose:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        root.addHandler(sh)

    _LOGGING_CONFIGURED = True


def add_file_handler(log_file, level: Optional[str] = None) -> None:
    """Attach a FileHandler to the root logger.

    Use this from an entry point to add a per-run log file once the output
    directory is known (i.e. after ``setup_logging`` has configured the console).
    Idempotent: does nothing if a FileHandler for the same resolved path is
    already attached.
    """
    log_path = Path(log_file)
    resolved = str(log_path.resolve())

    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler):
            if str(Path(handler.baseFilename).resolve()) == resolved:
                return

    os.makedirs(log_path.parent, exist_ok=True)

    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter(_LOG_FORMAT))
    if level:
        fh.setLevel(level.upper() if isinstance(level, str) else level)
    root.addHandler(fh)
