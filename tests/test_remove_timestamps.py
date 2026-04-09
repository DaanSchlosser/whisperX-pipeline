"""Tests for transcript timestamp cleanup helpers."""

from pathlib import Path
from tempfile import TemporaryDirectory

from remove_timestamps import default_output_path, remove_timestamps, strip_timestamp_prefix


def test_strip_timestamp_prefix_handles_timestamped_line() -> None:
    """Strips a leading WhisperX timestamp token when present."""
    line = "[00:00:01.234 -> 00:00:03.456]  SPEAKER_00: Hello\n"

    assert strip_timestamp_prefix(line) == "SPEAKER_00: Hello\n"


def test_strip_timestamp_prefix_preserves_plain_line() -> None:
    """Leaves lines unchanged when no timestamp prefix exists."""
    line = "SPEAKER_01: No timestamp here\n"

    assert strip_timestamp_prefix(line) == line


def test_default_output_path() -> None:
    """Derives the default output path with the expected suffix."""
    input_path = Path("Transcriptions/interview.txt")

    assert default_output_path(input_path) == Path("Transcriptions/interview_no_timestamps.txt")


def test_remove_timestamps_writes_cleaned_file() -> None:
    """Writes output with timestamp prefixes removed from each line."""
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        source = temp_path / "input.txt"
        target = temp_path / "output.txt"
        source.write_text(
            "[00:00:00.001 -> 00:00:01.000]  SPEAKER_00: A\nSPEAKER_01: B\n",
            encoding="utf-8",
        )

        remove_timestamps(source, target)

        assert target.read_text(encoding="utf-8") == "SPEAKER_00: A\nSPEAKER_01: B\n"
