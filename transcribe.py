import os
import sys
import time
import logging
import argparse
import platform
import subprocess
import warnings
from pathlib import Path

# Suppress noisy informational messages from third-party libraries
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=r"(?s).*torchcodec.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r"(?s).*TensorFloat-32.*")
warnings.filterwarnings("ignore", message=r"(?s).*degrees of freedom.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r"(?s).*Lightning automatically upgraded.*")

import whisperx
from whisperx.diarize import DiarizationPipeline
import torch
from dotenv import load_dotenv

load_dotenv()

# Ensure HF_TOKEN is available globally for huggingface_hub downloads
# (pyannote sub-models like PLDA read this env var directly)
if os.getenv("HF_TOKEN") and not os.getenv("HUGGING_FACE_HUB_TOKEN"):
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.getenv("HF_TOKEN")


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    print("WARNING: CUDA not available, falling back to CPU (this will be slow)")
    return "cpu"


def get_audio_duration(audio_path: str) -> float | None:
    """Get audio duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def print_system_info(device: str, compute_type: str):
    """Print hardware and system details."""
    print("-" * 60)
    print("SYSTEM INFO")
    print("-" * 60)
    print(f"  OS:          {platform.system()} {platform.release()}")
    print(f"  Python:      {platform.python_version()}")
    print(f"  PyTorch:     {torch.__version__}")
    print(f"  Device:      {device}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  GPU:         {gpu_name} ({vram:.1f} GB VRAM)")
        print(f"  CUDA:        {torch.version.cuda}")
    print(f"  Compute:     {compute_type}")
    print()


def print_file_info(audio_path: str):
    """Print audio file details."""
    p = Path(audio_path)
    size_mb = p.stat().st_size / (1024 ** 2)
    duration = get_audio_duration(audio_path)

    print("-" * 60)
    print("FILE INFO")
    print("-" * 60)
    print(f"  File:        {p.name}")
    print(f"  Format:      {p.suffix}")
    print(f"  Size:        {size_mb:.1f} MB")
    if duration:
        print(f"  Duration:    {format_duration(duration)}")
    print()


def step_timer(name: str):
    """Context manager that prints step name and elapsed time."""
    class Timer:
        def __init__(self):
            self.elapsed = 0.0
        def __enter__(self):
            self._start = time.perf_counter()
            print(f"  {name} ...", end="", flush=True)
            return self
        def __exit__(self, *_):
            self.elapsed = time.perf_counter() - self._start
            print(f" done ({format_duration(self.elapsed)})")
    return Timer()


def transcribe(audio_path: str, model_name: str = "large-v2", language: str = None,
               min_speakers: int = None, max_speakers: int = None,
               batch_size: int = 16, compute_type: str = "float16"):
    """Run the full WhisperX pipeline: transcribe → align → diarize."""

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        sys.exit("ERROR: HF_TOKEN not found. Create a .env file (see .env.example).")

    device = get_device()
    if device == "cpu":
        compute_type = "int8"

    audio_path = str(Path(audio_path).resolve())

    print_system_info(device, compute_type)
    print_file_info(audio_path)

    print("-" * 60)
    print(f"PIPELINE  (model={model_name})")
    print("-" * 60)

    total_start = time.perf_counter()

    # --- 1. Transcribe ---
    with step_timer("[1/4] Transcribing audio"):
        model = whisperx.load_model(model_name, device, compute_type=compute_type,
                                    language=language)
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=batch_size, language=language)
    detected_lang = result.get("language", language)
    print(f"         Detected language: {detected_lang}")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- 2. Align ---
    with step_timer("[2/4] Aligning timestamps"):
        align_model, metadata = whisperx.load_align_model(
            language_code=detected_lang, device=device
        )
        result = whisperx.align(
            result["segments"], align_model, metadata, audio, device,
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
        diarize_kwargs = {}
        if min_speakers is not None:
            diarize_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarize_kwargs["max_speakers"] = max_speakers
        diarize_segments = diarize_model(audio, **diarize_kwargs)

    # --- 4. Assign speakers ---
    with step_timer("[4/4] Assigning speakers"):
        result = whisperx.assign_word_speakers(diarize_segments, result)

    total_elapsed = time.perf_counter() - total_start
    print()
    print(f"  Total time: {format_duration(total_elapsed)}")

    return result, detected_lang


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def save_results(result: dict, audio_path: str, output_dir: str):
    """Save transcription as a .txt file."""
    stem = Path(audio_path).stem
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    segments = result.get("segments", [])

    txt_path = out / f"{stem}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for seg in segments:
            speaker = seg.get("speaker", "UNKNOWN")
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            text = seg["text"].strip()
            f.write(f"[{start} -> {end}]  {speaker}:  {text}\n")
    print(f"  TXT  → {txt_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe and diarize audio with WhisperX + pyannote"
    )
    parser.add_argument("audio", help="Path to the audio file")
    parser.add_argument("--model", default="large-v2",
                        help="Whisper model name (default: large-v2)")
    parser.add_argument("--language", default=None,
                        help="Language code, e.g. 'nl' or 'en' (auto-detect if omitted)")
    parser.add_argument("--min-speakers", type=int, default=None,
                        help="Minimum expected number of speakers")
    parser.add_argument("--max-speakers", type=int, default=None,
                        help="Maximum expected number of speakers")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for transcription (lower if GPU OOM)")
    parser.add_argument("--compute-type", default="float16",
                        choices=["float16", "float32", "int8"],
                        help="Compute type (default: float16)")
    parser.add_argument("--output-dir", default="Transcriptions",
                        help="Output directory (default: Transcriptions)")
    args = parser.parse_args()

    result, _ = transcribe(
        audio_path=args.audio,
        model_name=args.model,
        language=args.language,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        batch_size=args.batch_size,
        compute_type=args.compute_type,
    )

    print()
    print(f"Saving results ({len(result.get('segments', []))} segments) ...")
    save_results(result, args.audio, args.output_dir)
    print()
    print("Done!")


if __name__ == "__main__":
    main()
