import argparse
from pathlib import Path


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_no_timestamps{input_path.suffix}")


def strip_timestamp_prefix(line: str) -> str:
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
    with input_path.open("r", encoding="utf-8") as infile, output_path.open("w", encoding="utf-8") as outfile:
        for line in infile:
            outfile.write(strip_timestamp_prefix(line))


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove WhisperX timestamp prefixes from a transcript file")
    parser.add_argument("input", help="Path to the transcript .txt file")
    parser.add_argument("-o", "--output", help="Output file path (default: '<input stem>_no_timestamps.txt')")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else default_output_path(input_path)

    remove_timestamps(input_path, output_path)
    print(f"Timestamps removed. Output saved to {output_path}")


if __name__ == "__main__":
    main()