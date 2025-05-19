# Transcript Grading System (0-1000 Score)

This document proposes a scoring framework for evaluating ASR transcripts against a corrected reference.

## Overview
- **Purpose**: Provide a single numeric score (0–1000) to summarise transcript quality.
- **Inputs**:
  - Generated transcript(s) from the tool.
  - Reference transcript with manually corrected text.
  - Optional diarisation metadata.
- **Outputs**:
  - `overall_score` in the range 0–1000.
  - Breakdown of component sub-scores (WER, diarisation, language handling).

## Scoring Components
The overall score is composed of three parts:

1. **Phrase Accuracy (80% weight)**
   - Measured using Word Error Rate (WER) between the combined transcript and reference.
   - `phrase_score = max(0, 1 - WER) * 800`.
2. **Multilingual Handling (10% weight)**
   - Determines whether the system correctly transcribes lines in multiple languages.
   - Use `language_utils.identify_spanish_lines` (and similar heuristics for other languages) on both reference and predicted transcripts.
   - Compute the fraction of correctly identified multilingual lines and scale to 100 points.
   - **Gate**: If multilingual accuracy is below 50%, the final score cannot exceed 500.
3. **Diarisation Accuracy (10% weight)**
   - Optional speaker-turn accuracy if diarisation metadata is provided.
   - Compute F1 score of speaker boundaries compared with reference. Scale to 100 points.
   - If no diarisation info is available, allocate the full 100 points to phrase accuracy instead.

The `overall_score` is the sum of the three component scores, capped between 0 and 1000.  Systems that do not support multilingual transcription are capped at 500 regardless of other metrics.

## Example Calculation
```
WER = 0.15  # 15% word error
multilingual_accuracy = 0.85
speaker_f1 = 0.70

phrase_score      = (1 - 0.15) * 800 = 680
multilingual_score = 0.85 * 100       = 85
speaker_score      = 0.70 * 100       = 70

overall = phrase_score + multilingual_score + speaker_score
        = 835  (no cap triggered)
```

## Usage in Experiments
Whenever `transcribe_and_evaluate.py` is run, compute the three metrics and print the resulting score alongside WER.  Future scripts can log these values to compare models and configurations.

