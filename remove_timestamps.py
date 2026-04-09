"""Utilities to remove WhisperX timestamp prefixes from transcript files."""

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def default_output_path(input_path: Path) -> Path:
    """Return the default output path for a cleaned transcript."""
    return input_path.with_name(f"{input_path.stem}_no_timestamps{input_path.suffix}")


def strip_timestamp_prefix(line: str) -> str:
    """Remove a leading timestamp token from one line if present."""
    if not line.startswith("["):
        return line

    _, separator, remainder = line.partition("]  ")
    if separator:
        return remainder

    _, separator, remainder = line.partition("] ")
    if separator:
        return remainder

    return line


def remove_timestamps(input_path: Path, output_path: Path) -> None:
    """Write a copy of input transcript without timestamp prefixes."""
    with input_path.open("r", encoding="utf-8") as infile, output_path.open("w", encoding="utf-8") as outfile:
        for line in infile:
            outfile.write(strip_timestamp_prefix(line))


def main() -> None:
    """Parse CLI args and remove timestamps from a transcript file."""
    parser = argparse.ArgumentParser(description="Remove WhisperX timestamp prefixes from a transcript file")
    parser.add_argument("input", help="Path to the transcript .txt file")
    parser.add_argument("-o", "--output", help="Output file path (default: '<input stem>_no_timestamps.txt')")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else default_output_path(input_path)

    remove_timestamps(input_path, output_path)
    logger.info("Timestamps removed. Output saved to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    main()
