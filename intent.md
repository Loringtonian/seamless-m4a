# Intent Specification – Language Lesson Transcription Bot

## 1 Vision (Why does this project exist?)
Deliver an **offline, speaker-aware, multilingual transcription tool** that ingests `.m4a` Zoom recordings of one-to-one language lessons and produces rich artefacts (clean transcript, word-level JSON, embeddings) suitable for downstream grading by teachers and LLMs.

## 2 Stakeholders & Primary Goals
| Stakeholder | Intention |
|-------------|-----------|
| Language teacher (non-technical) | Drag & drop Zoom file → receive diarised, language-tagged transcript for grading. |
| Student | Receive timely, language-annotated feedback. |
| Developer / Researcher | Maintain pipeline; tune accuracy; run controlled experiments. |
| Offline MacBook Air (Intel) | Run everything locally without cloud calls. |

## 3 Functional Intent
1. **Input Handling**  
   • Accept `.m4a` (AAC) recordings.  
   • Convert to 16 kHz mono WAV.  
   • Chunk > 45 min into ≤ 15 min pieces.
2. **Optional VAD**  
   • Trim long silences using PyAnnote segmentation.
3. **ASR**  
   • Transcribe with `faster-whisper` (default `large-v3`).  
   • Return sentence-level segments + word timestamps (via WhisperX).
4. **Speaker Diarisation**  
   • Identify speaker turns using PyAnnote.  
   • Target ≤ 2 speakers (teacher + student) but handle extras.
5. **Speaker ⇄ Student Mapping**  
   • Map diarised IDs to codes in `students.json`.  
   • Expose each speaker's allowed languages list.
6. **Language Identification**  
   • Segment-level language tag with fastText `lid.176`.  
   • Override improbable predictions outside `speaker_allowed_langs`.  
   • (Optional) trigger UATMP tiny-model rescoring on low-confidence segments.
7. **Post-Processing Artefacts**  
   • Generate:
	- `clean.txt` — plain text with `[LANG]<CODE>` tags.  
	- `rich.json` — `{word,start_ms,end_ms,language,speaker}` array.  
	- `embeddings.parquet` — SentenceTransformer embeddings + metadata.  
	- `qualitative_comparison_report.txt` when reference is provided.
8. **Automated Evaluation**  
   • Compute WER vs `corrected_transcript.txt` when available.
9. **Logging & Experimentation**  
   • CLI `--log_level`, rotating file logs.  
   • `run_pipeline.sh` enforces Conda env and streams progress.

## 4 Non-Functional Intent
* Runs **off-grid** after first-time model download.  
* Single-machine, CPU-only (Intel macOS) within ~25 min real-time per 90 s clip (stretch: near-realtime for full lesson).  
* Accuracy targets: ≤ 25 % WER baseline, aspirational ≤ 0.5 % (roadmap).  
* Extensible to additional languages (initial: EN/ES/DE).  
* Uniform logging format for observability.

## 5 External Dependencies (intent-level)
* Conda env defined in `environment.yml`.  
* `ffmpeg` CLI on PATH.  
* Model assets cached under `./models/` and `$HF_HOME`.

## 6 Assumptions & Constraints
* Teacher provides `students.json` with codes + allowed languages.  
* Only one student and one teacher per recording (edge cases treated as UNKNOWN).  
* Reference transcript for WER is manually curated and can deviate in punctuation/casing.

## 7 Success / Done Definition
* Running `./run_pipeline.sh <audio>.m4a` produces the four artefacts in `output_runs/.../` without fatal errors.  
* WER, speaker mismatch count and qualitative score logged to `experiment_log.md`.  
* No internet required after `cache_models.py` run.

## 8 Out-of-Scope (for now)
* UI / Electron wrapper.  
* Real-time streaming transcription.  
* Automatic voice-print speaker recognition.  
* Word-level LID for German (pending future model selection).

---
This specification captures the *intent* of the project; individual module designs and algorithmic choices may evolve as long as they satisfy these high-level aims. 