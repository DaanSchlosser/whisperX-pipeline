"""WhisperX transcription pipeline with alignment and speaker diarization."""

from __future__ import annotations

import argparse
import functools
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Any

# These filters MUST run before the offending libraries import, so they stay at
# module top. warnings.filterwarnings uses re.match (anchored), so patterns match
# from the start of the message. torchcodec fires at import time inside
# pyannote.audio.core.io, so filtering by module is more robust than matching
# its multi-line message.
warnings.filterwarnings("ignore", module=r"pyannote\.audio\.core\.io")
warnings.filterwarnings("ignore", message=r"std\(\): degrees of freedom")
warnings.filterwarnings("ignore", message=r"TensorFloat-32")
warnings.filterwarnings("ignore", message=r"Lightning automatically upgraded")

import torch  # noqa: E402
import whisperx  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from whisperx.diarize import DiarizationPipeline  # noqa: E402

logger = logging.getLogger(__name__)

_current_step: list[str] = [""]  # mutable single-element avoids global statement

SECONDS_PER_MINUTE = 60
FFPROBE_TIMEOUT_SECONDS = 30
LANGUAGE_CODE_RE = re.compile(r"^[a-z]{2,10}$")


@dataclass(frozen=True)
class TranscriptionOptions:
    """User-controlled parameters for a single transcription run."""

    model_name: str = "large-v2"
    language: str | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    batch_size: int = 16
    compute_type: str = "float16"


@dataclass(frozen=True)
class TranscriptionResult:
    """Outcome of a transcription run: diarized segments plus source metadata."""

    segments: list[dict[str, Any]]
    language: str
    audio_path: Path
    duration_seconds: float | None


# ---------------------------------------------------------------------------
# Environment & logging setup
# ---------------------------------------------------------------------------


def configure_environment() -> None:
    """Load .env and quiet noisy third-party loggers."""
    load_dotenv()

    # pyannote sub-models read HUGGING_FACE_HUB_TOKEN directly via huggingface_hub.
    if (hf_token := os.getenv("HF_TOKEN")) and not os.getenv("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)


class _StepPrefixFilter(logging.Filter):
    """Prepend the active step label (e.g. ``[1/4]``) to third-party log lines."""

    def filter(self, record: logging.LogRecord) -> bool:
        if _current_step[0]:
            record.msg = f"{_current_step[0]} {record.msg}"
        return True


def configure_logging() -> None:
    """Install a single formatter across our logger and whisperx/pyannote loggers.

    Third-party libraries (whisperx especially) attach their own handlers with
    their own formats, and something in the dependency chain replays records
    routed through root — hence the mixed-format lines and duplicated output
    seen when using plain ``logging.basicConfig``. Attaching our handler
    directly to each logger we care about and disabling propagation prevents
    both the restyling and the replay.
    """
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    step_filter = _StepPrefixFilter()

    def _adopt(lg: logging.Logger, *, step_prefix: bool = False) -> None:
        lg.setLevel(logging.INFO)
        lg.handlers.clear()
        lg.addHandler(handler)
        lg.propagate = False
        if step_prefix:
            lg.addFilter(step_filter)

    _adopt(logger)
    for name in ("whisperx", "pyannote", "lightning.pytorch"):
        _adopt(logging.getLogger(name), step_prefix=True)


# ---------------------------------------------------------------------------
# ffprobe helpers
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _ffprobe_path() -> str | None:
    """Return the resolved ffprobe executable path, or None (warning logged once)."""
    path = shutil.which("ffprobe")
    if path is None:
        logger.warning("ffprobe was not found on PATH; audio metadata will be limited")
    return path


def _ffprobe_field(audio_path: Path, field_spec: str) -> str | None:
    """Query a single ffprobe field (e.g. ``format=duration``) for the audio file.

    Returns the trimmed stdout on success, or None on any failure. ffprobe is
    bounded by :data:`FFPROBE_TIMEOUT_SECONDS` so a pathological file cannot
    stall the pipeline indefinitely.
    """
    ffprobe = _ffprobe_path()
    if ffprobe is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603 -- ffprobe path resolved via shutil.which
            [
                ffprobe,
                "-v",
                "quiet",
                "-show_entries",
                field_spec,
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=FFPROBE_TIMEOUT_SECONDS,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None
    value = result.stdout.strip()
    return value or None


def get_audio_duration(audio_path: Path) -> float | None:
    """Return audio duration in seconds, or None if ffprobe is unavailable."""
    raw = _ffprobe_field(audio_path, "format=duration")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def get_recording_datetime(audio_path: Path) -> datetime | None:
    """Return best-effort recording datetime for the audio file.

    Prefers the container's ``creation_time`` tag (what recording devices write
    at capture time); falls back to filesystem creation time, then modification
    time. On Windows, ``st_ctime`` is creation time; on POSIX it is change time
    — so we prefer ``st_birthtime`` when the platform exposes it.
    """
    raw = _ffprobe_field(audio_path, "format_tags=creation_time")
    if raw is not None:
        try:
            return datetime.fromisoformat(raw).astimezone()
        except ValueError:
            pass

    try:
        stat = audio_path.stat()
    except OSError:
        return None
    ts = getattr(stat, "st_birthtime", None) or stat.st_ctime
    try:
        return datetime.fromtimestamp(ts).astimezone()
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_transcription_request(audio_path: Path, options: TranscriptionOptions) -> None:
    """Validate input path and option combinations before expensive model loading."""
    if not audio_path.exists() or not audio_path.is_file():
        msg = f"Audio file not found: {audio_path}"
        raise FileNotFoundError(msg)

    if options.batch_size < 1:
        msg = "--batch-size must be >= 1"
        raise ValueError(msg)

    if options.min_speakers is not None and options.min_speakers < 1:
        msg = "--min-speakers must be >= 1"
        raise ValueError(msg)

    if options.max_speakers is not None and options.max_speakers < 1:
        msg = "--max-speakers must be >= 1"
        raise ValueError(msg)

    if options.min_speakers and options.max_speakers and options.min_speakers > options.max_speakers:
        msg = "--min-speakers cannot be greater than --max-speakers"
        raise ValueError(msg)

    if options.language and not LANGUAGE_CODE_RE.fullmatch(options.language):
        msg = "--language must be an ISO-style code like 'en' or 'nl'"
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration."""
    if seconds < SECONDS_PER_MINUTE:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), SECONDS_PER_MINUTE)
    h, m = divmod(m, SECONDS_PER_MINUTE)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS.mmm notation."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------


def get_device() -> str:
    """Return the preferred Torch device."""
    if torch.cuda.is_available():
        return "cuda"
    logger.warning("CUDA not available, falling back to CPU (this will be slow)")
    return "cpu"


def log_system_info(device: str, compute_type: str) -> None:
    """Log hardware and system details."""
    info = [
        "-" * 60,
        "SYSTEM INFO",
        "-" * 60,
        f"  OS:          {platform.system()} {platform.release()}",
        f"  Python:      {platform.python_version()}",
        f"  PyTorch:     {torch.__version__}",
        f"  Device:      {device}",
    ]
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram: float = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        cuda_version = getattr(getattr(torch, "version", None), "cuda", "unknown")
        info.append(f"  GPU:         {gpu_name} ({vram:.1f} GB VRAM)")
        info.append(f"  CUDA:        {cuda_version}")
        info.append(f"  Compute:     {compute_type}")
    for line in info:
        logger.info("%s", line)


def log_file_info(audio_path: Path, duration_seconds: float | None) -> None:
    """Log audio file details."""
    size_mb = audio_path.stat().st_size / (1024**2)
    info = [
        "-" * 60,
        "FILE INFO",
        "-" * 60,
        f"  File:        {audio_path.name}",
        f"  Format:      {audio_path.suffix}",
        f"  Size:        {size_mb:.1f} MB",
    ]
    if duration_seconds is not None:
        info.append(f"  Duration:    {format_duration(duration_seconds)}")
    for line in info:
        logger.info("%s", line)


@contextmanager
def step_timer(name: str) -> Generator[None, None, None]:
    """Context manager that logs step name and elapsed time."""
    logger.info("%s ...", name)
    _current_step[0] = name.split("]", 1)[0] + "]"  # e.g. "[1/4]"
    start = time.perf_counter()
    try:
        yield
    finally:
        _current_step[0] = ""
        elapsed = time.perf_counter() - start
        logger.info("%s done (%s)", name, format_duration(elapsed))


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def transcribe(audio_path: Path, options: TranscriptionOptions) -> TranscriptionResult:
    """Run the full WhisperX pipeline: transcribe, align, diarize, assign speakers."""
    validate_transcription_request(audio_path, options)

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        msg = "HF_TOKEN not found. Create a .env file (see .env.example)."
        raise RuntimeError(msg)

    device = get_device()
    compute_type = options.compute_type
    if device == "cpu" and compute_type != "int8":
        logger.info("CPU device: overriding compute_type %s -> int8", compute_type)
        compute_type = "int8"

    resolved_path = audio_path.resolve()
    duration_seconds = get_audio_duration(resolved_path)

    log_system_info(device, compute_type)
    log_file_info(resolved_path, duration_seconds)

    logger.info("-" * 60)
    logger.info("PIPELINE  (model=%s)", options.model_name)
    logger.info("-" * 60)

    total_start = time.perf_counter()

    with step_timer("[1/4] Transcribing audio"):
        model = whisperx.load_model(
            options.model_name,
            device,
            compute_type=compute_type,
            language=options.language,
        )
        audio = whisperx.load_audio(str(resolved_path))
        result = model.transcribe(audio, batch_size=options.batch_size, language=options.language)
    detected_lang = result.get("language") or options.language
    if detected_lang is None:
        msg = "Unable to determine language automatically. Re-run with --language set explicitly."
        raise RuntimeError(msg)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    with step_timer("[2/4] Aligning timestamps"):
        align_model, metadata = whisperx.load_align_model(
            language_code=detected_lang,
            device=device,
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )

    del align_model
    if device == "cuda":
        torch.cuda.empty_cache()

    with step_timer("[3/4] Speaker diarization"):
        diarize_model = DiarizationPipeline(
            model_name="pyannote/speaker-diarization-3.1",
            token=hf_token,
            device=device,
        )
        diarize_kwargs: dict[str, Any] = {}
        if options.min_speakers is not None:
            diarize_kwargs["min_speakers"] = options.min_speakers
        if options.max_speakers is not None:
            diarize_kwargs["max_speakers"] = options.max_speakers
        diarize_segments = diarize_model(audio, **diarize_kwargs)

    with step_timer("[4/4] Assigning speakers"):
        result = whisperx.assign_word_speakers(diarize_segments, result)

    total_elapsed = time.perf_counter() - total_start
    logger.info("  Total time: %s", format_duration(total_elapsed))

    return TranscriptionResult(
        segments=result.get("segments", []),
        language=detected_lang,
        audio_path=resolved_path,
        duration_seconds=duration_seconds,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def count_speakers(segments: list[dict[str, Any]]) -> int:
    """Return the number of unique speakers in the diarized segments."""
    return len({seg["speaker"] for seg in segments if seg.get("speaker") and seg["speaker"] != "UNKNOWN"})


def build_metadata_header(
    result: TranscriptionResult,
    options: TranscriptionOptions,
) -> str:
    """Return a ``# Key: Value`` header block describing the transcription.

    The leading ``#`` prefix keeps the block invisible to the timestamp-stripping
    regex in ``remove_timestamps.py`` (which matches ``[HH:MM:SS.mmm -> ...]``
    prefixes only), so cleaned outputs retain the metadata unchanged.
    """
    recorded = get_recording_datetime(result.audio_path)
    generated = datetime.now().astimezone()

    rows: list[tuple[str, str]] = [("Source", result.audio_path.name)]
    if recorded is not None:
        rows.append(("Recorded", recorded.strftime("%Y-%m-%d %H:%M:%S")))
    if result.duration_seconds is not None:
        rows.append(("Duration", format_duration(result.duration_seconds)))
    rows.extend(
        [
            ("Language", result.language),
            ("Model", options.model_name),
            ("Speakers", str(count_speakers(result.segments))),
            ("Segments", str(len(result.segments))),
            ("Generated", generated.strftime("%Y-%m-%d %H:%M:%S")),
        ]
    )

    key_width = max(len(k) for k, _ in rows)
    return "\n".join(f"# {k:<{key_width}}  {v}" for k, v in rows) + "\n\n"


def save_results(
    result: TranscriptionResult,
    output_dir: Path,
    options: TranscriptionOptions,
) -> Path:
    """Write transcription as a .txt file with a metadata header; return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / f"{result.audio_path.stem}.txt"
    header = build_metadata_header(result, options)

    with txt_path.open("w", encoding="utf-8") as f:
        f.write(header)
        for seg in result.segments:
            speaker: str = seg.get("speaker", "UNKNOWN")
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            text: str = seg["text"].strip()
            f.write(f"[{start} -> {end}]  {speaker}:  {text}\n")

    logger.info("  TXT  -> %s", txt_path)
    return txt_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe and diarize audio with WhisperX + pyannote",
    )
    parser.add_argument("audio", help="Path to the audio file")
    parser.add_argument(
        "--model",
        default="large-v2",
        help="Whisper model name (default: large-v2)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code, e.g. 'nl' or 'en' (auto-detect if omitted)",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Minimum expected number of speakers",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Maximum expected number of speakers",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for transcription (lower if GPU OOM; try 4 on 8GB VRAM)",
    )
    parser.add_argument(
        "--compute-type",
        default="float16",
        choices=["float16", "float32", "int8"],
        help="Compute type (default: float16; auto-overridden to int8 on CPU)",
    )
    parser.add_argument(
        "--output-dir",
        default="Transcriptions",
        help="Output directory (default: Transcriptions)",
    )
    return parser


def main() -> None:
    """Parse CLI args and run transcription."""
    configure_environment()
    args = _build_arg_parser().parse_args()

    options = TranscriptionOptions(
        model_name=args.model,
        language=args.language,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        batch_size=args.batch_size,
        compute_type=args.compute_type,
    )

    try:
        result = transcribe(audio_path=Path(args.audio), options=options)
        logger.info("Saving results (%d segments) ...", len(result.segments))
        save_results(result, Path(args.output_dir), options)
        logger.info("Done!")
    except (FileNotFoundError, ValueError, RuntimeError):
        logger.exception("Transcription failed")
        sys.exit(2)


if __name__ == "__main__":
    configure_logging()
    main()
