# Experiment Instructions

This document explains how to run parameter sweeps with `transcribe_and_evaluate.py`.

1. Choose parameters to vary (e.g. `--duration`, `--device`).
2. Run the script for each combination:

```
python transcribe_and_evaluate.py --student recordings/Student\ Audio.m4a \
    --teacher recordings/Teacher\ Audio.m4a \
    --reference "corrected transcript.txt" \
    --duration 80 --device cpu
```

3. After each run a row is appended to `experiment_log.md`.
4. Review the log to see how each change impacts WER and Spanish detection accuracy.

Add additional columns or parameters as needed.
