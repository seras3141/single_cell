"""
Root conftest: mock heavy optional dependencies so the test suite can run
without a GPU environment (no torch/cellpose needed for unit tests).
"""
import sys
from types import ModuleType
from unittest.mock import MagicMock


def _ensure_mock(name: str) -> None:
    """Insert a MagicMock into sys.modules for *name* and all sub-packages."""
    if name not in sys.modules:
        mock = MagicMock()
        mock.__name__ = name
        mock.__spec__ = None
        sys.modules[name] = mock


for _pkg in ("torch", "torchvision", "torchaudio", "cellpose"):
    _ensure_mock(_pkg)
    # Pre-register common sub-modules so dotted imports resolve too
    for _sub in ("nn", "cuda", "optim", "utils", "plot", "models", "core"):
        _ensure_mock(f"{_pkg}.{_sub}")
