# Calibrated AI-Text Detection

A calibration + fairness-audit layer over one open-weight AI-text detector.

> **Result goes here first.** Once M2 lands this README leads with the
> before/after reliability diagram and the ECE-collapse number (target ~0.25 ->
> <0.05), not the architecture. The layering is an appendix detail.

## What this is

Open AI-text detectors emit an overconfident, uncalibrated score that no
benchmark measures for reliability or subgroup fairness. This project:

1. **Calibrates** one open detector (RoBERTa, whose raw logit we own) to a
   probability with a known error rate - ECE/Brier + reliability diagram on a
   group-wise-split MAGE pool.
2. **Audits** its native-vs-non-native false-positive gap with bootstrap
   confidence intervals on the Stanford TOEFL set, including a deliberately weak
   log-perplexity detector as a positive control.

## Status

- [x] M1 - `DetectorPort` + RoBERTa adapter emitting raw logits (smoke-tested).
- [ ] M2 - Calibrator + reliability diagram (ECE before/after) on MAGE.
- [ ] M3 - Fairness audit: FPR gap + bootstrap CIs on TOEFL, weak control.

Future work (deliberately cut): third-party API adapter, per-subgroup ECE,
conformal abstention, serving API, a fine-tuned model.

## Setup

```bash
uv sync --extra dev
uv run pytest -s        # runs the M1 smoke test; prints raw logits
```

First run downloads `openai-community/roberta-base-openai-detector`.

## Layout

```
src/aitcal/
  detectors/port.py      # DetectorPort protocol (the one interface)
  detectors/roberta.py   # RoBERTa adapter -> raw logit
  data/sample.py         # tiny labeled sample for the smoke test
tests/test_smoke.py      # M1 deliverable
```
