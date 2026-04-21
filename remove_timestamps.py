"""Utilities to remove WhisperX timestamp prefixes from transcript files."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)
TIMESTAMP_PREFIX_RE = re.compile(r"^\[\d{2}:\d{2}:\d{2}\.\d{3}\s*->\s*\d{2}:\d{2}:\d{2}\.\d{3}\]\s*")


@dataclass(frozen=True)
class CleanupStats:
    """Summary metrics for a timestamp-cleanup run."""

    total_lines: int
    stripped_lines: int


def default_output_path(input_path: Path) -> Path:
    """Return the default output path for a cleaned transcript."""
    return input_path.with_name(f"{input_path.stem}_no_timestamps{input_path.suffix}")


def strip_timestamp_prefix(line: str) -> str:
    """Remove a leading timestamp token from one line if present."""
    return TIMESTAMP_PREFIX_RE.sub("", line, count=1)


def validate_paths(input_path: Path, output_path: Path) -> None:
    """Validate input/output paths before processing."""
    if not input_path.exists() or not input_path.is_file():
        msg = f"Input transcript not found: {input_path}"
        raise FileNotFoundError(msg)

    if input_path.resolve() == output_path.resolve():
        msg = "Output path must be different from input path"
        raise ValueError(msg)


def remove_timestamps(input_path: Path, output_path: Path) -> CleanupStats:
    """Write a copy of input transcript without timestamp prefixes.

    Writes to a sibling ``*.tmp`` file and atomically renames on success so a
    partial file cannot be mistaken for a complete one if the run is interrupted.
    """
    validate_paths(input_path, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    total_lines = 0
    stripped_lines = 0
    try:
        with input_path.open(encoding="utf-8") as infile, tmp_path.open("w", encoding="utf-8") as outfile:
            for line in infile:
                total_lines += 1
                cleaned = strip_timestamp_prefix(line)
                if cleaned != line:
                    stripped_lines += 1
                outfile.write(cleaned)
        tmp_path.replace(output_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    return CleanupStats(total_lines=total_lines, stripped_lines=stripped_lines)


def main() -> None:
    """Parse CLI args and remove timestamps from a transcript file."""
    parser = argparse.ArgumentParser(description="Remove WhisperX timestamp prefixes from a transcript file")
    parser.add_argument("input", help="Path to the transcript .txt file")
    parser.add_argument("-o", "--output", help="Output file path (default: '<input stem>_no_timestamps.txt')")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else default_output_path(input_path)

    try:
        stats = remove_timestamps(input_path, output_path)
    except (FileNotFoundError, ValueError, OSError):
        logger.exception("Timestamp cleanup failed")
        sys.exit(2)

    logger.info(
        "Timestamps removed. Output saved to %s (stripped %d/%d lines)",
        output_path,
        stats.stripped_lines,
        stats.total_lines,
    )


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    main()
