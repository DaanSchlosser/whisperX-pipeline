"""WhisperX transcription pipeline with alignment and speaker diarization."""

from __future__ import annotations

import argparse
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
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Any

import torch
import whisperx
from dotenv import load_dotenv
from whisperx.diarize import DiarizationPipeline

logger = logging.getLogger(__name__)

# Suppress noisy informational messages from third-party libraries
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=r"(?s).*torchcodec.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r"(?s).*TensorFloat-32.*")
warnings.filterwarnings("ignore", message=r"(?s).*degrees of freedom.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r"(?s).*Lightning automatically upgraded.*")

load_dotenv()

# Ensure HF_TOKEN is available globally for huggingface_hub downloads
# (pyannote sub-models like PLDA read this env var directly)
if (_hf_token := os.getenv("HF_TOKEN")) and not os.getenv("HUGGING_FACE_HUB_TOKEN"):
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token

LANGUAGE_CODE_RE = re.compile(r"^[a-z]{2,10}$")


@dataclass(frozen=True)
class TranscriptionOptions:
    """Runtime options for a single transcription run."""

    model_name: str = "large-v2"
    language: str | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    batch_size: int = 16
    compute_type: str = "float16"


def validate_transcription_request(audio_path: str, options: TranscriptionOptions) -> None:
    """Validate input path and option combinations before expensive model loading."""
    audio_file = Path(audio_path)
    if not audio_file.exists() or not audio_file.is_file():
        msg = f"Audio file not found: {audio_file}"
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


def get_device() -> str:
    """Return the preferred Torch device."""
    if torch.cuda.is_available():
        return "cuda"
    logger.warning("CUDA not available, falling back to CPU (this will be slow)")
    return "cpu"


def get_audio_duration(audio_path: str) -> float | None:
    """Get audio duration in seconds using ffprobe."""
    audio_path_obj = Path(audio_path)
    if not audio_path_obj.exists() or not audio_path_obj.is_file():
        return None

    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        logger.warning("ffprobe was not found on PATH; skipping duration detection")
        return None

    try:
        result = subprocess.run(  # noqa: S603 -- shell is disabled and command path is resolved via shutil.which
            [
                ffprobe_path,
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path_obj),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except subprocess.CalledProcessError:
        return None


SECONDS_PER_MINUTE = 60


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < SECONDS_PER_MINUTE:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), SECONDS_PER_MINUTE)
    h, m = divmod(m, SECONDS_PER_MINUTE)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def print_system_info(device: str, compute_type: str) -> str:
    """Return hardware and system details as a string."""
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
    system_info = "\n".join(info)
    logger.info("\n%s", system_info)
    return system_info


def print_file_info(audio_path: str) -> str:
    """Return audio file details as a string."""
    p = Path(audio_path)
    size_mb = p.stat().st_size / (1024**2)
    duration = get_audio_duration(audio_path)
    info = [
        "-" * 60,
        "FILE INFO",
        "-" * 60,
        f"  File:        {p.name}",
        f"  Format:      {p.suffix}",
        f"  Size:        {size_mb:.1f} MB",
    ]
    if duration:
        info.append(f"  Duration:    {format_duration(duration)}")
    file_info = "\n".join(info)
    logger.info("\n%s", file_info)
    return file_info


@contextmanager
def step_timer(name: str) -> Generator[None, None, None]:
    """Context manager that logs step name and elapsed time."""
    logger.info("%s ...", name)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("%s done (%s)", name, format_duration(elapsed))


def transcribe(
    audio_path: str,
    options: TranscriptionOptions,
) -> tuple[dict[str, Any], str]:
    """Run the full WhisperX pipeline: transcribe, align, diarize."""
    validate_transcription_request(audio_path, options)

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        msg = "ERROR: HF_TOKEN not found. Create a .env file (see .env.example)."
        sys.exit(msg)

    device = get_device()
    compute_type = options.compute_type
    if device == "cpu":
        compute_type = "int8"

    audio_path = str(Path(audio_path).resolve())

    print_system_info(device, compute_type)
    print_file_info(audio_path)

    logger.info("-" * 60)
    logger.info("PIPELINE  (model=%s)", options.model_name)
    logger.info("-" * 60)

    total_start = time.perf_counter()

    # --- 1. Transcribe ---
    with step_timer("[1/4] Transcribing audio"):
        model = whisperx.load_model(
            options.model_name,
            device,
            compute_type=compute_type,
            language=options.language,
        )
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=options.batch_size, language=options.language)
    detected_lang = result.get("language") or options.language
    if detected_lang is None:
        msg = "Unable to determine language automatically. Re-run with --language set explicitly."
        raise RuntimeError(msg)
    logger.info("         Detected language: %s", detected_lang)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- 2. Align ---
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

    # --- 3. Diarize ---
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

    # --- 4. Assign speakers ---
    with step_timer("[4/4] Assigning speakers"):
        result = whisperx.assign_word_speakers(diarize_segments, result)

    total_elapsed = time.perf_counter() - total_start
    logger.info("")
    logger.info("  Total time: %s", format_duration(total_elapsed))

    return result, detected_lang


def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS.mmm notation."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def save_results(result: dict[str, Any], audio_path: str, output_dir: str) -> None:
    """Save transcription as a .txt file."""
    stem = Path(audio_path).stem
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    segments: list[dict[str, Any]] = result.get("segments", [])

    txt_path = out / f"{stem}.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        for seg in segments:
            speaker: str = seg.get("speaker", "UNKNOWN")
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            text: str = seg["text"].strip()
            f.write(f"[{start} -> {end}]  {speaker}:  {text}\n")
    logger.info("  TXT  -> %s", txt_path)


def main() -> None:
    """Parse CLI args and run transcription."""
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
        help="Batch size for transcription (lower if GPU OOM)",
    )
    parser.add_argument(
        "--compute-type",
        default="float16",
        choices=["float16", "float32", "int8"],
        help="Compute type (default: float16)",
    )
    parser.add_argument(
        "--output-dir",
        default="Transcriptions",
        help="Output directory (default: Transcriptions)",
    )
    args = parser.parse_args()

    options = TranscriptionOptions(
        model_name=args.model,
        language=args.language,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        batch_size=args.batch_size,
        compute_type=args.compute_type,
    )

    try:
        result, _ = transcribe(
            audio_path=args.audio,
            options=options,
        )

        logger.info("")
        logger.info("Saving results (%d segments) ...", len(result.get("segments", [])))
        save_results(result, args.audio, args.output_dir)
        logger.info("")
        logger.info("Done!")
    except (FileNotFoundError, ValueError, RuntimeError):
        logger.exception("Transcription failed")
        sys.exit(2)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    main()
