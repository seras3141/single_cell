"""Tests for src/utils/logging_utils.py."""

import logging
import sys
from pathlib import Path

import pytest

from src.utils import logging_utils
from src.utils.logging_utils import setup_logging


def _reset_logging():
    """Remove all handlers and clear the idempotency guard so each test starts clean."""
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    logging_utils._LOGGING_CONFIGURED = False


@pytest.fixture(autouse=True)
def clean_logging():
    """Reset root logger before and after every test."""
    _reset_logging()
    yield
    _reset_logging()


class TestSetupLogging:
    def test_default_adds_stream_handler(self):
        setup_logging()
        root = logging.getLogger()
        assert any(isinstance(h, logging.StreamHandler) for h in root.handlers)

    def test_level_is_applied(self):
        setup_logging(level="WARNING")
        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_level_string_uppercased(self):
        setup_logging(level="debug")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_no_stream_handler_when_not_verbose(self):
        setup_logging(verbose=False)
        root = logging.getLogger()
        # Use exact type check to exclude pytest's LogCaptureHandler (a StreamHandler subclass)
        stream_handlers = [h for h in root.handlers if type(h) is logging.StreamHandler]
        assert len(stream_handlers) == 0

    def test_file_handler_created(self, tmp_path):
        log_file = tmp_path / "logs" / "test.log"
        setup_logging(log_file=log_file)
        root = logging.getLogger()
        assert any(isinstance(h, logging.FileHandler) for h in root.handlers)
        assert log_file.parent.exists()

    def test_log_file_directory_created(self, tmp_path):
        log_file = tmp_path / "nested" / "dir" / "app.log"
        assert not log_file.parent.exists()
        setup_logging(log_file=log_file)
        assert log_file.parent.exists()

    def test_log_config_level_used(self):
        setup_logging(level=None, log_config={"level": "ERROR"})
        root = logging.getLogger()
        assert root.level == logging.ERROR

    def test_explicit_level_overrides_log_config(self):
        setup_logging(level="WARNING", log_config={"level": "DEBUG"})
        root = logging.getLogger()
        assert root.level == logging.WARNING
