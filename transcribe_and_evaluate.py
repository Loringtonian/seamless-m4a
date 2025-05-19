#!/usr/bin/env python
"""Simple transcription + evaluation tool using Meta SeamlessM4T v2 ASR.

Usage (run from repo root):
    python transcribe_and_evaluate.py \
        --student '/Users/lts/Desktop/seamless m4a/recordings/Student Audio.m4a' \
        --teacher '/Users/lts/Desktop/seamless m4a/recordings/Teacher Audio.m4a' \
        --reference '/Users/lts/Desktop/seamless m4a/corrected transcript.txt' \
        --duration 80

This script will:
1. Convert both m4a files to 16-kHz, mono WAV trimmed to the first `--duration` seconds.
2. Transcribe the trimmed WAVs with the `facebook/seamless-m4t-v2-large` model (CPU-only by default).
3. Save individual transcripts to `<basename>.txt` in the working directory.
4. Concatenate the two transcripts with speaker tags into `combined_transcript.txt`.
5. Compute Word Error Rate (WER) between the combined transcript (first `--duration` seconds) and the user-supplied corrected transcript using `jiwer`.

Note: The current implementation performs a naïve concatenation (STUDENT first, then TEACHER).  A more
accurate interleaving based on segment timestamps will be added in future iterations.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List

import torch
import torchaudio

# Transformers 4.38+ exposes SeamlessM4Tv2ForSpeechToText. Provide graceful fallback if running older version.
try:
    from transformers import SeamlessM4Tv2ForSpeechToText
except ImportError:  # pragma: no cover
    from transformers import SeamlessM4TModel as SeamlessM4Tv2ForSpeechToText  # type: ignore

from transformers import AutoProcessor, logging as hf_logging
from jiwer import wer
from jiwer import Compose, RemovePunctuation, ToLowerCase, RemoveMultipleSpaces, Strip

# Suppress HF generation warnings
hf_logging.set_verbosity_error()

def convert_to_wav(
    in_path: Path, out_path: Path, start_sec: int = 0, duration_sec: int | None = None
) -> None:
    """Convert (and optionally trim) any audio file to 16-kHz mono WAV using ffmpeg."""
    ffmpeg_cmd: List[str] = [
        "ffmpeg",
        "-y",  # overwrite
        "-i",
        str(in_path),
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
    ]
    if start_sec:
        ffmpeg_cmd.extend(["-ss", str(start_sec)])
    if duration_sec:
        ffmpeg_cmd.extend(["-t", str(duration_sec)])
    ffmpeg_cmd.append(str(out_path))

    # Run ffmpeg quietly
    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def transcribe_with_seamless(audio_path: Path, device: str = "cpu") -> str:
    """Transcribe `audio_path` (16-kHz WAV) with SeamlessM4T v2 ASR and return text."""
    model_name = "facebook/seamless-m4t-v2-large"

    processor = AutoProcessor.from_pretrained(model_name)
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(model_name).to(device)
    model.eval()

    # Load waveform
    waveform, sr = torchaudio.load(str(audio_path))  # (channels, time)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000

    # SeamlessM4T expects shape (time,) or (time, channels)?? We'll flatten to mono.
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Convert to float32
    waveform = waveform.to(torch.float32)

    inputs = processor(audios=waveform.squeeze(0), return_tensors="pt", sampling_rate=sr)

    # decoder prompt: task="transcribe" with no explicit language forces ASR to keep the source language(s)
    decoder_input_ids = processor.get_decoder_prompt_ids(task="transcribe")

    generated_tokens = model.generate(**inputs, decoder_input_ids=decoder_input_ids)
    # Decode (skip special tokens like <|en|>)
    transcription: str = processor.decode(generated_tokens[0], skip_special_tokens=True)
    return transcription.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe two speaker tracks with SeamlessM4T and evaluate WER.")
    parser.add_argument("--student", type=Path, required=True, help="Path to Student Audio .m4a file")
    parser.add_argument("--teacher", type=Path, required=True, help="Path to Teacher Audio .m4a file")
    parser.add_argument("--reference", type=Path, required=True, help="Path to corrected transcript text file")
    parser.add_argument("--duration", type=int, default=80, help="Max seconds of audio to process (default: 80)")
    parser.add_argument("--device", type=str, default="cpu", help="Device for model inference: 'cpu' or 'cuda'")
    args = parser.parse_args()

    # Verify ffmpeg exists
    if subprocess.call(["which", "ffmpeg"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        raise EnvironmentError("ffmpeg not found. Please install ffmpeg and ensure it's on PATH.")

    # Prepare temp WAVs
    with tempfile.TemporaryDirectory() as tmpdir:
        student_wav = Path(tmpdir) / "student.wav"
        teacher_wav = Path(tmpdir) / "teacher.wav"
        convert_to_wav(args.student, student_wav, start_sec=0, duration_sec=args.duration)
        convert_to_wav(args.teacher, teacher_wav, start_sec=0, duration_sec=args.duration)

        # Transcribe
        print("Transcribing Student track …")
        student_text = transcribe_with_seamless(student_wav, device=args.device)
        print("Student transcript:\n", student_text, "\n", sep="")

        print("Transcribing Teacher track …")
        teacher_text = transcribe_with_seamless(teacher_wav, device=args.device)
        print("Teacher transcript:\n", teacher_text, "\n", sep="")

    # Save transcripts
    Path("student_transcript.txt").write_text(student_text, encoding="utf-8")
    Path("teacher_transcript.txt").write_text(teacher_text, encoding="utf-8")

    # Combine (naïve order: Student then Teacher)
    combined_text = f"STUDENT: {student_text}\nTEACHER: {teacher_text}"
    Path("combined_transcript.txt").write_text(combined_text, encoding="utf-8")

    # Reference text (only first duration seconds available)
    reference_text = args.reference.read_text(encoding="utf-8").strip()

    # Compute WER with basic normalization (remove punctuation, lowercase, etc.)
    transformation = Compose([RemovePunctuation(), ToLowerCase(), RemoveMultipleSpaces(), Strip()])

    error_rate = wer(
        reference_text,
        combined_text,
        truth_transform=transformation,
        hypothesis_transform=transformation,
    )
    print(f"\nWord Error Rate (WER) vs reference (first {args.duration} s): {error_rate * 100:.2f}%")


if __name__ == "__main__":
    main() 