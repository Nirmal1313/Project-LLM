"""File reading module."""

from __future__ import annotations

from pathlib import Path

from src.tokenizer.core.constants import Defaults
from src.tokenizer.core.exceptions import FileReadError
from src.tokenizer.core.logging import LoggerMixin


class FileReader(LoggerMixin):
    
    def __init__(self, encoding: str = Defaults.FILE_ENCODING) -> None:
        """
        Initialize the file reader.
        
        Args:
            encoding: Character encoding to use (default: utf-8)
        """
        self._encoding = encoding
    
    def read(self, file_path: Path) -> str:
        file_path = Path(file_path)  # Ensure it's a Path object
        
        if not file_path.exists():
            raise FileReadError(
                f"File not found: {file_path}",
                file_path=str(file_path),
            )
        
        if not file_path.is_file():
            raise FileReadError(
                f"Path is not a file: {file_path}",
                file_path=str(file_path),
            )
        
        try:
            self.logger.debug(f"Reading file: {file_path}")
            with open(file_path, "r", encoding=self._encoding) as f:
                content = f.read()
            self.logger.info(f"Read {len(content)} characters from {file_path.name}")
            return content
        except UnicodeDecodeError as e:
            raise FileReadError(
                f"Unicode decode error in file: {file_path}",
                file_path=str(file_path),
                original_error=e,
            )
        except PermissionError as e:
            raise FileReadError(
                f"Permission denied: {file_path}",
                file_path=str(file_path),
                original_error=e,
            )
        except Exception as e:
            raise FileReadError(
                f"Error reading file: {file_path}",
                file_path=str(file_path),
                original_error=e,
            )
    
    def __repr__(self) -> str:
        return f"FileReader(encoding={self._encoding!r})"
