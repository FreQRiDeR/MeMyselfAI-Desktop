"""
logger.py
Centralized logging setup for MeMyselfAI.
"""

import logging
import sys
from backend.redact import redact, RedactingFilter
from PyQt6.QtCore import QObject, pyqtSignal


class QtLogHandler(logging.Handler, QObject):
    """
    A logging handler that emits a Qt signal for each log record.
    This allows the UI to display logs in real-time.
    It also caches records until the UI is ready to consume them.
    """
    new_log_record = pyqtSignal(logging.LogRecord)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        QObject.__init__(self)
        self.record_cache = []

    def emit(self, record):
        """Emit the log record via a Qt signal and cache it."""
        # Create a copy to avoid modifying the original record
        redacted_record = logging.makeLogRecord(record.__dict__)
        redacted_record.msg = redact(redacted_record.msg)
        if isinstance(redacted_record.args, (list, tuple)):
            redacted_record.args = tuple(redact(arg) for arg in redacted_record.args)

        self.record_cache.append(redacted_record)
        self.new_log_record.emit(redacted_record)

    def get_cache(self):
        """Return all cached records."""
        return list(self.record_cache)


# Global instance of the handler
qt_log_handler = QtLogHandler()


def setup_logging(log_level: str = "INFO"):
    """
    Configure the root logger to output to console and the Qt handler.

    Args:
        log_level: The minimum level of logs to capture (e.g., "DEBUG", "INFO").
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Basic formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove any existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(RedactingFilter())
    root_logger.addHandler(console_handler)

    # Qt handler
    qt_log_handler.setFormatter(formatter)
    qt_log_handler.addFilter(RedactingFilter())
    root_logger.addHandler(qt_log_handler)

    logging.info(f"Logging initialized with level {log_level}")

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module."""
    return logging.getLogger(name)