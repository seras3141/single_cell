#!/usr/bin/env python3
"""Thin repository wrapper for the threshold activity labeler CLI."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cell_activity_labeler.__main__ import main


if __name__ == "__main__":
    main()