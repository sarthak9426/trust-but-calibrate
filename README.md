# Calibrated AI-Text Detection

A calibration + fairness-audit layer over one open-weight AI-text detector.

## Result

The RoBERTa AI-text detector is badly overconfident. On a leakage-audited,
group-wise-split slice of MAGE it reports a confidence that is nowhere near its
real accuracy - **ECE 0.448**. A Platt calibrator fit on held-out sources
collapses that to **ECE 0.041** (Brier 0.421 -> 0.123), turning the raw score
into a probability you can trust.

![Reliability diagram: raw RoBERTa vs Platt-calibrated on MAGE](docs/reliability.png)

The red curve is the raw detector: it sits far above the diagonal, so a "10%
machine" score really means ~85% machine. The green curve is calibrated: it
hugs the diagonal.

One finding worth its own line: **temperature scaling is not enough here.** The
reflexive fix (a single scalar T) only reaches ECE 0.34, because this slice is
class-imbalanced and needs a *shift*, not just a rescale. Platt and isotonic
(which learn a shift) both clear ECE < 0.05.

Reproduce:

```bash
uv sync --extra dev
uv run python -m aitcal.scripts.run_m2 --limit 1500   # default calibrator: platt
```

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
- [x] M2 - Calibrator (temperature / Platt / isotonic) + reliability diagram;
      ECE 0.448 -> 0.041 on a group-wise-split MAGE pool.
- [ ] M3 - Fairness audit: FPR gap + bootstrap CIs on TOEFL, weak control.

Future work (deliberately cut): third-party API adapter, per-subgroup ECE,
conformal abstention, serving API, a fine-tuned model.

## Setup

```bash
uv sync --extra dev
uv run pytest            # M1 smoke + M2 unit tests
```

First run of the M2 script downloads `openai-community/roberta-base-openai-detector`
and a MAGE slice.

## Layout

```
src/aitcal/
  detectors/port.py         # DetectorPort protocol (the one interface)
  detectors/roberta.py      # RoBERTa adapter -> raw logit
  calibration/calibrator.py # temperature / Platt / isotonic
  eval/metrics.py           # ECE, Brier
  eval/reliability.py       # before/after reliability diagram
  data/mage.py              # MAGE loader + group-wise split + leakage audit
  data/sample.py            # tiny labeled sample for the M1 smoke test
  scripts/run_m2.py         # end-to-end M2: score -> split -> calibrate -> report
tests/                      # smoke (M1) + calibration unit tests (M2)
```
