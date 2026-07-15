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
        if name == "torch":
            # scipy's array_api_compat calls issubclass(cls, torch.Tensor) at import
            # time (e.g. via trackpy -> scipy.stats); a MagicMock attribute is not a
            # valid issubclass argument, so give Tensor a real (if empty) class.
            mock.Tensor = type("Tensor", (), {})
        sys.modules[name] = mock


for _pkg in ("torch", "torchvision", "torchaudio", "cellpose"):
    _ensure_mock(_pkg)
    # Pre-register common sub-modules so dotted imports resolve too
    for _sub in ("nn", "cuda", "optim", "utils", "plot", "models", "core"):
        _ensure_mock(f"{_pkg}.{_sub}")

# scipy's array-api-compat runs `issubclass(cls, torch.Tensor)` at import time. A bare
# MagicMock attr isn't a class, so that raises TypeError and breaks collection of any
# test that (transitively) imports scipy. Give the torch mock a real, empty Tensor
# class so the check degrades to False cleanly.
if isinstance(sys.modules.get("torch"), MagicMock):
    sys.modules["torch"].Tensor = type("Tensor", (), {})
