# Seamless M4A – Simple Transcription & Evaluation

This repository provides a **minimal working example** of how to:

1. Transcribe bilingual/tri-lingual lesson recordings with **Meta's SeamlessM4T-v2** ASR model.
2. Compute the **word-error-rate (WER)** of the transcript against a manually-corrected reference.

The implementation is intentionally lightweight – it runs fully offline on CPU (given enough RAM) and can be extended later with speaker diarisation, language identification, etc.

---

## 1. Set-up

1. **Install system prerequisites**
   ```bash
   # macOS (Homebrew example)
   brew install ffmpeg
   ```

2. **Create a Python environment & install packages**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   *PyTorch* will be installed with CPU support by default.  If you have a CUDA-capable GPU, install the matching `torch` wheel for increased speed and set `--device cuda` when running the script.

---

## 2. Usage

```bash
python transcribe_and_evaluate.py \
  --student   "/Users/<user>/Desktop/seamless m4a/recordings/Student Audio.m4a" \
  --teacher   "/Users/<user>/Desktop/seamless m4a/recordings/Teacher Audio.m4a" \
  --reference "/Users/<user>/Desktop/seamless m4a/corrected transcript.txt" \
  --duration 80            # Seconds of audio to process (default 80)
```

The script will:

1. Convert the two `.m4a` files to temporary 16-kHz mono WAVs trimmed to the first *n* seconds (80 by default).
2. Run SeamlessM4T-v2 ASR and save:  
   * `student_transcript.txt`  
   * `teacher_transcript.txt`  
   * `combined_transcript.txt` (naïve concatenation with speaker tags)
3. Print the **WER** versus the provided reference text (first *n* seconds only).
4. Append a summary to `experiment_log.md` with the CLI options, runtime,
   word counts, current Git hash and WER.

> **Note**: For now the merging of speaker tracks is a simple concatenation (student first, then teacher).  In future iterations we'll use timestamps to interleave utterances chronologically.

---

## 3. Next steps

*   Replace naïve speaker-merge with timestamp-based interleaving.
*   Add optional **faster-whisper** fallback for low-RAM machines.
*   Support *automatic* language diarisation for more accurate code-switch evaluation.
*   Package the script as a reusable CLI (`pipx install`). 