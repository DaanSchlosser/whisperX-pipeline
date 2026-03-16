# WhisperX Transcription Pipeline

Local audio transcription with speaker diarization using [WhisperX](https://github.com/m-bain/whisperX) and [pyannote](https://github.com/pyannote/pyannote-audio).

## Prerequisites

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) on your PATH
- A [HuggingFace](https://huggingface.co) account with access accepted for:
  - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  - [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
  - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

## Setup

```bash
pip install -r requirements.txt
```

**GPU (recommended):** If pip installed a CPU-only PyTorch, reinstall with CUDA:

```bash
# For NVIDIA GPUs (RTX 30xx/40xx/50xx — CUDA 12.6)
pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu126
```

> **Older GPUs** (GTX 10xx series): use `--compute-type int8` instead of the default `float16` since Pascal GPUs don't support fp16 in ctranslate2.

Then create your `.env` file:

```bash
cp .env.example .env
# Edit .env and paste your HuggingFace token
```

## Usage

```bash
python transcribe.py <audio_file> [options]
```

### Default (recommended for interviews)

```bash
python transcribe.py "Audio/interview.mp3" --max-speakers 3
```

### All options

| Flag | Default | Description |
|---|---|---|
| `--model` | `large-v2` | Whisper model name |
| `--language` | auto-detect | Language code (`nl`, `en`, etc.) |
| `--min-speakers` | — | Minimum number of speakers |
| `--max-speakers` | — | Maximum number of speakers |
| `--batch-size` | `16` | Transcription batch size (lower if OOM) |
| `--compute-type` | `float16` | `float16`, `float32`, or `int8` |
| `--output-dir` | `Transcriptions` | Output directory |

### Examples

```bash
# Dutch interview, 2 speakers
python transcribe.py "Audio/interview.mp3" --language nl --max-speakers 2

# English meeting, force int8 for older GPUs or CPU
python transcribe.py "Audio/meeting.wav" --language en --max-speakers 5 --compute-type int8

# CPU-only (int8 is used automatically)
python transcribe.py "Audio/file.mp3" --batch-size 4
```

## Output

Results are saved to `Transcriptions/` as a `.txt` file with timestamps and speaker labels per line:

```
[00:00:01.234 -> 00:00:05.678]  SPEAKER_00:  Hello, how are you?
[00:00:06.100 -> 00:00:09.200]  SPEAKER_01:  I'm doing well, thanks.
```

## License

MIT — see [LICENSE](LICENSE).

## GPU vs CPU

| | GPU (CUDA) | CPU |
|---|---|---|
| Speed | ~10x–50x faster | Slow |
| Compute type | `float16` (RTX 20xx+) or `int8` (GTX 10xx) | `int8` (auto) |
| Batch size | 8–16 | 4 |
| VRAM needed | ~6–8 GB for `large-v2` | — |
