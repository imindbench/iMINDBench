"""Atomic result persistence and shared resume checks."""

import json
import os
import tempfile
from pathlib import Path


def is_valid_result_file(file_path: str | Path) -> bool:
    """Accept readable JSON objects, including results from older releases."""
    try:
        with open(file_path, encoding="utf-8") as handle:
            return isinstance(json.load(handle), dict)
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return False


def write_result_json(results: dict, file_path: str | Path) -> None:
    """Publish a complete JSON file without exposing an intermediate write."""
    destination = Path(file_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        # A sibling file keeps replacement atomic on the destination filesystem.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(results, handle, indent=4)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
