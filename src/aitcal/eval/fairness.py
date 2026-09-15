"""Fairness metrics: false-positive rate per group and the gap between groups.

On this data every essay is HUMAN, so the only error a detector can make is a
FALSE POSITIVE (calling human text machine). The fairness question is whether
that FPR is higher for non-native writers than native ones.

  FPR(group) = fraction of that group's human essays flagged machine
  gap        = FPR(non-native) - FPR(native)

n ~ 91 per group is too small for a subgroup reliability diagram, so we report
the gap with a BOOTSTRAP confidence interval (resample within each group), which
is valid at this n. A CI that excludes 0 is a gap the data supports.
"""

from __future__ import annotations

import numpy as np


def false_positive_rate(pred_machine: np.ndarray) -> float:
    """FPR on all-human data = fraction predicted machine. pred_machine in {0,1}."""
    pred = np.asarray(pred_machine, dtype=int)
    return float(pred.mean()) if len(pred) else float("nan")


def fpr_gap_bootstrap(
    pred_nonnative: np.ndarray,
    pred_native: np.ndarray,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Bootstrap the FPR gap = FPR(non-native) - FPR(native).

    Resamples each group independently with replacement. Returns point estimates
    and the percentile CI on the gap.
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
        "n_nonnative": int(len(nn)),
        "n_native": int(len(na)),
    }
