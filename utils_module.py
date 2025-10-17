from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Union


class Utils:
    """Utility helpers for file operations, logging, and JSON I/O."""

    @staticmethod
    def validate_path(path: Union[str, Path], create: bool = False) -> Path:
        """Validate a filesystem path and optionally create it."""
        resolved = Path(path)
        if create:
            resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    @staticmethod
    def setup_logging(log_dir: Union[str, Path], name: str = "experiment") -> logging.Logger:
        """Configure a logger that writes to disk and stdout."""
        log_dir_path = Utils.validate_path(log_dir, create=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir_path / f"{name}_{timestamp}.log"

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        return logger

    @staticmethod
    def save_json(data: Any, path: Union[str, Path]) -> None:
        """Persist a Python object to disk as JSON."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w") as handle:
            json.dump(data, handle, indent=2)

    @staticmethod
    def load_json(path: Union[str, Path]) -> Any:
        """Load JSON data from disk, returning None if the file is missing."""
        source = Path(path)
        if not source.exists():
            return None
        with source.open("r") as handle:
            return json.load(handle)

    @staticmethod
    def read_file(path: Union[str, Path]) -> str:
        """Read text from a file."""
        with Path(path).open("r") as handle:
            return handle.read()

    @staticmethod
    def write_file(path: Union[str, Path], content: str) -> None:
        """Write text to a file, creating parent folders if needed."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w") as handle:
            handle.write(content)

    @staticmethod
    def delete_path(path: Union[str, Path]) -> None:
        """Remove a file or directory tree."""
        target = Path(path)
        if target.is_file():
            target.unlink()
        elif target.is_dir():
            import shutil

            shutil.rmtree(target)


def main() -> None:
    """Basic smoke test for Utils helpers."""
    temp_dir = Utils.validate_path("tmp_utils_test", create=True)
    Utils.save_json({"status": "ok"}, temp_dir / "test.json")
    loaded = Utils.load_json(temp_dir / "test.json")
    print(f"Loaded JSON content: {loaded}")
    Utils.delete_path(temp_dir)


if __name__ == "__main__":
    main()

