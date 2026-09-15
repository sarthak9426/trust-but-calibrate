"""M3 unit tests - fast, synthetic, no model or dataset download."""

from __future__ import annotations

import numpy as np

from aitcal.eval.fairness import false_positive_rate, fpr_gap_bootstrap


def test_fpr_basic() -> None:
    # 3 of 4 flagged machine on all-human data -> FPR 0.75.
    assert false_positive_rate(np.array([1, 1, 1, 0])) == 0.75
    assert false_positive_rate(np.array([0, 0, 0, 0])) == 0.0


def test_gap_sign_and_point_estimates() -> None:
    # non-native flagged 80%, native flagged 20% -> gap +0.6.
    nn = np.array([1] * 8 + [0] * 2)
    na = np.array([1] * 2 + [0] * 8)
    res = fpr_gap_bootstrap(nn, na, n_boot=2000, seed=0)
    assert abs(res["fpr_nonnative"] - 0.8) < 1e-9
    assert abs(res["fpr_native"] - 0.2) < 1e-9
    assert abs(res["gap"] - 0.6) < 1e-9


def test_ci_excludes_zero_on_strong_gap() -> None:
    # A large, consistent gap should give a CI clear of 0.
    nn = np.array([1] * 45 + [0] * 5)  # 90%
    na = np.array([1] * 5 + [0] * 45)  # 10%
    res = fpr_gap_bootstrap(nn, na, n_boot=5000, seed=0)
    assert res["excludes_zero"]
    assert res["ci_low"] > 0


def test_ci_includes_zero_on_no_gap() -> None:
    # Same FPR in both groups -> gap ~0, CI should straddle 0.
    rng = np.random.default_rng(1)
    nn = (rng.uniform(size=60) < 0.3).astype(int)
    na = (rng.uniform(size=60) < 0.3).astype(int)
    res = fpr_gap_bootstrap(nn, na, n_boot=5000, seed=0)
    assert not res["excludes_zero"]
    assert res["ci_low"] < 0 < res["ci_high"]
