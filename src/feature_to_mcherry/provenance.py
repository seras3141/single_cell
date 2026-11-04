"""Run provenance written beside the results of the feature_to_mcherry pipelines."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

logger = logging.getLogger(__name__)

PROVENANCE_FILENAME = "run_provenance.json"


def _git_commit(repo_root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _git_dirty(
    directory: Path, *, tracked_only: bool = False, paths: Sequence[str] = ()
) -> Optional[bool]:
    """Whether ``git status`` reports changes (``paths`` are repo-top pathspecs)."""
    command = ["git", "status", "--porcelain"]
    if tracked_only:
        command.append("--untracked-files=no")
    if paths:
        command += ["--", *paths]
    try:
        result = subprocess.run(
            command,
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip())


def write_run_provenance(config: Any, output_dir: Path) -> Optional[Path]:
    """Write the config, git commit, SLURM job id and time to ``output_dir``.

    A failure is logged and returns ``None``: it must not abort a finished run.
    """
    try:
        return _write_run_provenance(config, Path(output_dir))
    except Exception:
        logger.warning("Could not write %s", PROVENANCE_FILENAME, exc_info=True)
        return None


def _write_run_provenance(config: Any, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    package_dir = Path(__file__).resolve().parent
    record: Dict[str, Any] = {
        "config": (
            dataclasses.asdict(config)
            if dataclasses.is_dataclass(config)
            else {"repr": repr(config)}
        ),
        "git_commit": _git_commit(package_dir),
        "git_dirty": _git_dirty(package_dir, tracked_only=True),
        "git_dirty_code": _git_dirty(package_dir, paths=[":/src", ":/scripts"]),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = output_dir / PROVENANCE_FILENAME
    path.write_text(json.dumps(record, indent=2, default=str))
    return path
