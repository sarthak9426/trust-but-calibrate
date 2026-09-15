"""Fairness metrics: false-positive rate per group and the gap between groups.

On this data every essay is HUMAN, so the only error a detector can make is a
FALSE POSITIVE (calling human text machine). The fairness question is whether
that FPR is higher for non-native writers than native ones.

  FPR(group) = fraction of that group's human essays flagged machine
  gap        = FPR(non-native) - FPR(native)

n ~ 91 per group is too small for a subgroup reliability diagram, so we report
the gap with a BOOTSTRAP confidence interval (resample within each group), which
is valid at this n. A CI that excludes 0 is a gap the data supports.

When the decision threshold is FIT FROM THE DATA (e.g. anchored to the native
group's FPR), the cut is itself a sample statistic, so its variance must enter
the CI. `fpr_gap_bootstrap_scores` recomputes the threshold inside every
resample for exactly this reason; the prediction-based `fpr_gap_bootstrap` is
correct only for a FIXED, externally-given threshold.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def false_positive_rate(pred_machine: np.ndarray) -> float:
    """FPR on all-human data = fraction predicted machine. pred_machine in {0,1}."""
    pred = np.asarray(pred_machine, dtype=int)
    return float(pred.mean()) if len(pred) else float("nan")


def _summary(fpr_nn, fpr_na, gap, boot_gaps, ci, n_nn, n_na) -> dict:
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot_gaps, [alpha, 1.0 - alpha])
    return {
        "fpr_nonnative": fpr_nn,
        "fpr_native": fpr_na,
        "gap": gap,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci_level": ci,
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_nonnative": int(n_nn),
        "n_native": int(n_na),
    }


def fpr_gap_bootstrap(
    pred_nonnative: np.ndarray,
    pred_native: np.ndarray,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Bootstrap the FPR gap = FPR(non-native) - FPR(native) from FIXED predictions.

    Correct only when the machine/human threshold was given externally. When the
    threshold is fit from the data, use fpr_gap_bootstrap_scores instead.
    """
    rng = np.random.default_rng(seed)
    nn = np.asarray(pred_nonnative, dtype=int)
    na = np.asarray(pred_native, dtype=int)

    fpr_nn = false_positive_rate(nn)
    fpr_na = false_positive_rate(na)
    gap = fpr_nn - fpr_na

    boot_gaps = np.empty(n_boot)
    for i in range(n_boot):
        bs_nn = rng.choice(nn, size=len(nn), replace=True).mean()
        bs_na = rng.choice(na, size=len(na), replace=True).mean()
        boot_gaps[i] = bs_nn - bs_na

    return _summary(fpr_nn, fpr_na, gap, boot_gaps, ci, len(nn), len(na))


def fpr_gap_bootstrap_scores(
    scores_nonnative: np.ndarray,
    scores_native: np.ndarray,
    threshold_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Bootstrap the FPR gap with a DATA-FIT threshold recomputed per resample.

    threshold_fn(nn_scores, na_scores) -> cut. On the point estimate it runs on
    the full scores; inside each bootstrap iteration it runs on the RESAMPLED
    scores, so threshold-selection variance flows into the CI. FPR is the
    fraction of (human) scores strictly above the cut.
    """
    rng = np.random.default_rng(seed)
    nn = np.asarray(scores_nonnative, dtype=float)
    na = np.asarray(scores_native, dtype=float)

    def fprs(nn_s, na_s):
        cut = threshold_fn(nn_s, na_s)
        return float((nn_s > cut).mean()), float((na_s > cut).mean())

    fpr_nn, fpr_na = fprs(nn, na)
    gap = fpr_nn - fpr_na

    boot_gaps = np.empty(n_boot)
    for i in range(n_boot):
        bs_nn = rng.choice(nn, size=len(nn), replace=True)
        bs_na = rng.choice(na, size=len(na), replace=True)
        f_nn, f_na = fprs(bs_nn, bs_na)
        boot_gaps[i] = f_nn - f_na

    return _summary(fpr_nn, fpr_na, gap, boot_gaps, ci, len(nn), len(na))
