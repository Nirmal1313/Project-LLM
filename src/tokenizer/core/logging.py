"""Centralized logging configuration."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


# Default format for log messages
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Default log file settings
DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_FILE = "tokenizer.log"
DEFAULT_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
DEFAULT_BACKUP_COUNT = 3


def setup_logging(
    level: int = logging.INFO,
    format_string: str = LOG_FORMAT,
    date_format: str = DATE_FORMAT,
    stream: Optional[object] = None,
    log_to_file: bool = True,
    log_dir: str | Path = DEFAULT_LOG_DIR,
    log_file: str = DEFAULT_LOG_FILE,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
) -> Path | None:
        log_dir: Directory for log files (created if doesn't exist)
        log_file: Name of the log file
        max_bytes: Maximum size of log file before rotation (default 5MB)
        backup_count: Number of backup files to keep (default 3)
        
    Returns:
        Path to the log file if file logging is enabled, None otherwise
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(format_string, datefmt=date_format)
    
    # Console handler
    console_handler = logging.StreamHandler(stream or sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    log_file_path: Path | None = None
    
    # File handler with rotation
    if log_to_file:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir_path / log_file
        
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    return log_file_path


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__ or class name)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class LoggerMixin:
    """
    Mixin class that provides a logger property.
    
    Usage:
        class MyClass(LoggerMixin):
            def do_something(self):
                self.logger.info("Doing something")
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger bound to class name."""
        return get_logger(self.__class__.__name__)


# Module-level logger for this file
_logger = get_logger(__name__)


def log_function_call(func_name: str, **kwargs) -> None:
    """Log a function call with its arguments."""
    args_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    _logger.debug(f"Calling {func_name}({args_str})")
