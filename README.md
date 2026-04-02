# WhisperX Transcription Pipeline

Transcribe local audio with speaker diarization using [WhisperX](https://github.com/m-bain/whisperX) and [pyannote](https://github.com/pyannote/pyannote-audio).

## Requirements

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) on PATH
- Hugging Face account with access to:
  - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  - [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
  - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

## Setup (Windows, recommended)

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# CUDA-enabled PyTorch compatible with whisperx==3.8.x
.\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126

# FFmpeg
winget install --id Gyan.FFmpeg -e --accept-package-agreements --accept-source-agreements
```

Create `.env` from `.env.example` and set `HF_TOKEN`.

## Verify CUDA

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
```

## Usage

```powershell
.\.venv\Scripts\python.exe transcribe.py "Audio/interview.mp3" --max-speakers 3
```

If you get GPU out-of-memory (common on 8 GB VRAM):

```powershell
.\.venv\Scripts\python.exe transcribe.py "Audio/interview.mp3" --max-speakers 3 --batch-size 4 --compute-type int8
```

## Options

| Flag | Default | Description |
|---|---|---|
| `--model` | `large-v2` | Whisper model name |
| `--language` | auto-detect | Language code (`nl`, `en`, etc.) |
| `--min-speakers` | — | Minimum expected speakers |
| `--max-speakers` | — | Maximum expected speakers |
| `--batch-size` | `16` | Transcription batch size |
| `--compute-type` | `float16` | `float16`, `float32`, or `int8` |
| `--output-dir` | `Transcriptions` | Output directory |

## Output

Writes a `.txt` file to `Transcriptions/` with timestamps and speaker labels.

## Removing Timestamps

The `remove_timestamps.py` script strips timestamp prefixes from transcript files, in case you don't want them:

```powershell
# Auto-generates output filename
.\.venv\Scripts\python.exe remove_timestamps.py "Transcriptions/interview.txt"
# Output: Transcriptions/interview_no_timestamps.txt

# Specify custom output path
.\.venv\Scripts\python.exe remove_timestamps.py "Transcriptions/interview.txt" -o "interview_clean.txt"
```

This removes prefixes like `[00:00:01.234 -> 00:00:02.345]  ` from each line, converting this:
```
[00:00:00.151 -> 00:00:01.631]  SPEAKER_00: Good morning.
[00:00:01.651 -> 00:00:03.411]  SPEAKER_01: Hi there!
```

To:
```
SPEAKER_00: Good morning.
SPEAKER_01: Hi there!
```
