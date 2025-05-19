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
import subprocess
import tempfile
from pathlib import Path
from typing import List
import sys

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
import logging
import time
from datetime import datetime

import re

from language_utils import identify_spanish_lines
from grading import compute_transcript_score

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


def split_sentences(text: str) -> List[str]:
    """Very naive sentence splitter based on punctuation."""
    segments = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in segments if s.strip()]


def word_count(text: str) -> int:
    """Return the number of word tokens in ``text``."""
    return len(re.findall(r"\w+", text))


def log_experiment_results(
    *,
    args: argparse.Namespace,
    wer_value: float,
    student_time: float,
    teacher_time: float,
    total_time: float,
    student_words: int,
    teacher_words: int,
    git_hash: str,
    python_version: str,
) -> None:
    """Append a markdown table row with experiment metadata."""
    log_path = Path("experiment_log.md")
    if not log_path.exists():
        header = (
            "# Experiment Log\n\n"
            "| Timestamp | Student | Teacher | Duration (s) | Device | Git | Python | "
            "Student Words | Teacher Words | Student Time (s) | Teacher Time (s) | "
            "Total Time (s) | WER (%) |\n"
            "|-----------|---------|---------|--------------|--------|-----|--------|" \
            "--------------|--------------|------------------|------------------|" \
            "--------------|---------|\n"
        )
        log_path.write_text(header, encoding="utf-8")

    timestamp = datetime.utcnow().isoformat(timespec="seconds")
    row = (
        f"| {timestamp} | {args.student.name} | {args.teacher.name} | "
        f"{args.duration} | {args.device} | {git_hash} | {python_version} | "
        f"{student_words} | {teacher_words} | {student_time:.2f} | {teacher_time:.2f} | "
        f"{total_time:.2f} | {wer_value * 100:.2f} |\n"
    )
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe two speaker tracks with SeamlessM4T and evaluate WER.")
    parser.add_argument("--student", type=Path, required=True, help="Path to Student Audio .m4a file")
    parser.add_argument("--teacher", type=Path, required=True, help="Path to Teacher Audio .m4a file")
    parser.add_argument("--reference", type=Path, required=True, help="Path to corrected transcript text file")
    parser.add_argument("--duration", type=int, default=80, help="Max seconds of audio to process (default: 80)")
    parser.add_argument("--device", type=str, default="cpu", help="Device for model inference: 'cpu' or 'cuda'")
    parser.add_argument(
        "--experiment-log",
        type=Path,
        default=Path("experiment_log.md"),
        help="Path to markdown log where results will be appended",
    )
    # Variables below can be tuned in an experiment grid to improve performance.
    # For example:
    #   * --duration: clip length to transcribe
    #   * --device: 'cpu' or 'cuda'
    #   * start time or ASR model variant (requires code changes)
    # Add new parameters here to automatically log their impact.
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    run_start = time.perf_counter()

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
        logging.info("Transcribing Student track …")
        t0 = time.perf_counter()
        student_text = transcribe_with_seamless(student_wav, device=args.device)
        student_time = time.perf_counter() - t0
        logging.info("Student transcript:\n%s\n", student_text)
        logging.info("Student transcription time: %.2fs", student_time)

        logging.info("Transcribing Teacher track …")
        t0 = time.perf_counter()
        teacher_text = transcribe_with_seamless(teacher_wav, device=args.device)
        teacher_time = time.perf_counter() - t0
        logging.info("Teacher transcript:\n%s\n", teacher_text)
        logging.info("Teacher transcription time: %.2fs", teacher_time)

    # Save transcripts
    Path("student_transcript.txt").write_text(student_text, encoding="utf-8")
    Path("teacher_transcript.txt").write_text(teacher_text, encoding="utf-8")

    # Combine (naïve order: Student then Teacher). We also split each transcript
    # into simple sentence-like units so that language identification can be
    # applied per line instead of on the entire block.
    student_lines = split_sentences(student_text)
    teacher_lines = split_sentences(teacher_text)
    combined_lines = [f"STUDENT: {l}" for l in student_lines] + [f"TEACHER: {l}" for l in teacher_lines]
    combined_text = "\n".join(combined_lines)
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
    logging.info(
        "Word Error Rate (WER) vs reference (first %s s): %.2f%%",
        args.duration,
        error_rate * 100,
    )

    total_time = time.perf_counter() - run_start
    student_words = word_count(student_text)
    teacher_words = word_count(teacher_text)
    git_hash = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode()
        .strip()
    )
    python_version = sys.version.split()[0]

    logging.info(
        "Run summary: duration=%ss device=%s student_time=%.2fs teacher_time=%.2fs total_time=%.2fs WER=%.2f%% student_words=%s teacher_words=%s",
        args.duration,
        args.device,
        student_time,
        teacher_time,
        total_time,
        error_rate * 100,
        student_words,
        teacher_words,
    )

    log_experiment_results(
        args=args,
        wer_value=error_rate,
        student_time=student_time,
        teacher_time=teacher_time,
        total_time=total_time,
        student_words=student_words,
        teacher_words=teacher_words,
        git_hash=git_hash,
        python_version=python_version,
    )

    # Evaluate Spanish line detection
    ref_lines = [l.strip() for l in reference_text.splitlines() if l.strip()]
    pred_lines = [l.strip() for l in combined_text.splitlines() if l.strip()]
    ref_spanish = set(identify_spanish_lines(ref_lines))
    pred_spanish = set(identify_spanish_lines(pred_lines))
    accuracy = 0.0
    if ref_spanish:
        multilingual_accuracy = len(ref_spanish & pred_spanish) / len(ref_spanish)
    else:
        multilingual_accuracy = 1.0
    logging.info(
        "Spanish line detection accuracy: %.2f%%",
        multilingual_accuracy * 100,
    )

    overall = compute_transcript_score(
        wer=error_rate,
        multilingual_accuracy=multilingual_accuracy,
        speaker_f1=None,
    )
    logging.info("Overall transcript score: %.1f/1000", overall)

    # Append results to experiment log for later comparison
    log_path = args.experiment_log
    header = "| duration | device | WER | SpanishAcc |\n"
    separator = "|---|---|---|---|\n"
    if not log_path.exists():
        log_path.write_text("# Experiment Log\n\n" + header + separator, encoding="utf-8")

    spanish_acc = accuracy * 100 if ref_spanish else 0.0
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"| {args.duration} | {args.device} | {error_rate*100:.2f}% | {spanish_acc:.2f}% |\n")


if __name__ == "__main__":
    main() 
