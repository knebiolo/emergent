import sys
import io
import logging
from emergent.fish_passage.utils.logging import get_logger, safe_log_exception, configure_console_handler


def test_get_logger_returns_logger():
    logger = get_logger("test_logger", level=logging.DEBUG)
    assert isinstance(logger, logging.Logger)
    assert logger.level == logging.DEBUG


def test_safe_log_exception_with_logger(caplog):
    logger = get_logger("test_safe_logger")
    caplog.set_level(logging.ERROR)
    try:
        raise ValueError("test error")
    except Exception:
        safe_log_exception(logger, "caught error")
    # Ensure something was logged at ERROR level
    assert any(record.levelname == "ERROR" or record.levelname == "CRITICAL" or record.levelname == "EXCEPTION" for record in caplog.records)


def test_safe_log_exception_fallback_to_stderr(monkeypatch, capsys):
    # Simulate no logger provided
    try:
        raise RuntimeError("fallback")
    except Exception:
        safe_log_exception(None, "fallback message")
    captured = capsys.readouterr()
    assert "fallback message" in captured.err or "Traceback" in captured.err


def test_configure_console_handler_adds_handler():
    logger = logging.getLogger("cfg_logger")
    # Clear handlers
    logger.handlers = []
    configure_console_handler(logger, fmt="%(levelname)s:%(message)s", level=logging.INFO)
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
