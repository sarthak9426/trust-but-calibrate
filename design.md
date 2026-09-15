# Calibrated AI-Text Detection: a calibration + fairness audit layer

Status: DESIGN v2, de-risked after a tech-lead self-critique (2026-09-15).
Author: Sarthak (sarthak9426).

## The greenlit version (build THIS, not the original superset)

Build a **calibration-and-audit layer for ONE open-weight detector** (where we
own the raw logit), and ship exactly two defensible results:

(a) **Calibration result** - before/after reliability diagram + ECE/Brier on a
    group-wise-split MAGE-or-RAID pool, proving an overconfident detector was
    turned into a calibrated one (target headline: ECE ~0.25 -> <0.05).
(b) **One honest fairness audit** - native-vs-non-native FPR gap with BOOTSTRAP
    CONFIDENCE INTERVALS on the TOEFL set, framed as "here is the gap and its
    uncertainty for this detector," and deliberately including a WEAK/old
    detector so the method visibly catches a gap that exists.

The single result that sells this to an ML hiring manager: one reliability
diagram showing ECE collapse on a detector people trust blindly, paired with a
subgroup FPR gap with CIs. Concrete ML result + the judgment to measure fairness
honestly.

## Why these five things were CUT (to future work)

Each was a real risk the critique surfaced:

- **Third-party API adapter** - a "% AI" from GPTZero/Originality is opaque,
  already sigmoid-squashed, and silently retrained; temperature scaling on a
  non-logit is meaningless. Calibrate ONLY an open detector whose logit we own.
- **Per-subgroup ECE on TOEFL** - ~91 essays gives ~5 samples/bin, so a subgroup
  reliability diagram is noise. Report FPR gap with bootstrap CIs instead (valid
  at n~91); reserve ECE for the large MAGE/RAID pools only.
- **Conformal abstention** - deferring the low-confidence band "raises accuracy
  on the rest" by construction (a tautology) unless tied to a stated conformal
  risk level. Cut unless done as selective prediction with a risk guarantee.
- **Serving API** - nice systems polish, zero added ML signal for the portfolio.
- **Fine-tuned OwnModel** - a whole second project; the harness justifies it
  later, but it is not the sprint.

## Problem (one sentence)

Open AI-text detectors output an overconfident, uncalibrated score that no
benchmark measures for reliability or subgroup fairness; this project calibrates
one such detector to a probability with a known error rate and audits its
native-vs-non-native false-positive gap with confidence intervals.

## Novelty (verified 2026-09-14 landscape scan)

- No general calibration + fairness-audit layer for AI-text detectors exists.
  Cite/differentiate: MCP (ACL 2025, arXiv:2505.05084, conformal-for-FPR, not
  calibration) and Markov-Informed Calibration (arXiv:2602.08031, token-level
  score refinement, not confidence calibration).
- No benchmark (RAID, MAGE) reports ECE or a subgroup fairness gap. That is the
  contribution.
- Fairness nuance to encode HONESTLY: the non-native bias is real but shrinking,
  perplexity-mediated, language-dependent (2026 Czech replication; English TOEFL
  FPR ~61% -> ~23% on modern detectors). Deliberately include a weak detector so
  the audit demonstrably catches a gap; on a modern detector a small CI-bounded
  gap is a valid result, not a failure.

## Data plan (verified available; leakage-audited)

- Fairness axis: Stanford TOEFL non-native set (Liang 2023) + 2026 revisit
  (arXiv:2602.05769). FPR gap with bootstrap CIs (NOT subgroup ECE).
- Calibration pool: MAGE (27 generators) or RAID (11 models). ECE/Brier here,
  where n supports 10-15 bins.
- LEAKAGE AUDIT is a first-class harness output: group-wise split by
  source/generator/document family; fit calibration WITHIN-source only; emit
  overlap hashes so a document or generator in both calib and test is caught.
  (Cross-source contamination silently inflates every metric otherwise.)

## Architecture (kept minimal; result-first, not layering-first)

Still a clean `DetectorPort` so a second detector (and, later, our own) drops in,
but the README leads with the reliability diagram + the ECE number, NOT the
ports/adapters diagram. Layering is an appendix detail, not the pitch.

```
text -> DetectorPort.score(text) -> raw logit   [ONE open detector: RoBERTa or Fast-DetectGPT]
     -> Calibrator (temperature / Platt / isotonic; fit on held-out calib split)
     -> EvaluationHarness:
          calibration: ECE, Brier, reliability diagram (netcal) on MAGE/RAID
          fairness:    native-vs-non-native FPR gap + bootstrap CIs on TOEFL
          leakage:     group-wise split + overlap-hash audit
```

## Reusable libraries (do NOT reimplement)

scikit-learn (Platt, isotonic, Brier, calibration_curve), netcal (temperature
scaling, ECE, reliability diagrams), plus a bootstrap CI (scipy/numpy).

## Milestones (core = M1-M3; rest is explicit future work)

- M1 DetectorPort + one open HF detector; raw logits on a small labeled set.
- M2 Calibrator + reliability diagram; show raw detector miscalibrated,
    calibrated one not (ECE before/after) on a group-wise-split MAGE/RAID pool.
- M3 Fairness audit: native-vs-non-native FPR gap + bootstrap CIs on TOEFL,
    including a weak detector to prove the method catches a real gap.
- (FUTURE) conformal selective prediction with a risk guarantee; serving API;
  second/third detector adapters; fine-tuned OwnModel proven on the same harness.

## Open questions to resolve at build

- Which open detector first: RoBERTa openai-community (simple, owns logit) vs
  Fast-DetectGPT (stronger, needs a scoring LLM). Lean RoBERTa for M1 simplicity.
- MAGE vs RAID for the calibration pool (RAID larger + adversarial; MAGE simpler).
- Which weak/old detector to include as the fairness positive-control.
