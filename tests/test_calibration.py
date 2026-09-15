"""M2 unit tests - fast, synthetic, no model or dataset download.

They verify the pieces that carry the headline: ECE/Brier are correct on
hand-checkable inputs, a calibrator lowers ECE on a deliberately overconfident
signal, and the group-wise split never leaks a group across calib/test.
"""

from __future__ import annotations

import numpy as np

from aitcal.calibration import CALIBRATORS, raw_sigmoid
from aitcal.data.mage import Pool, group_wise_split
from aitcal.eval import brier, expected_calibration_error


def test_ece_zero_when_perfectly_calibrated() -> None:
    # Predictions that exactly match empirical frequency -> ECE 0.
    # 100 items at p=0.7, 70 of them positive.
    probs = np.full(100, 0.7)
    labels = np.array([1] * 70 + [0] * 30)
    assert expected_calibration_error(probs, labels, n_bins=15) < 1e-9


def test_ece_detects_miscalibration() -> None:
    # All predicted p=0.9 but only half are positive -> ECE ~ 0.4.
    probs = np.full(100, 0.9)
    labels = np.array([1] * 50 + [0] * 50)
    ece = expected_calibration_error(probs, labels, n_bins=15)
    assert abs(ece - 0.4) < 1e-9


def test_brier_matches_hand_value() -> None:
    probs = np.array([1.0, 0.0, 1.0, 0.0])
    labels = np.array([1, 0, 1, 0])
    assert brier(probs, labels) == 0.0
    probs2 = np.array([0.5, 0.5])
    labels2 = np.array([1, 0])
    assert abs(brier(probs2, labels2) - 0.25) < 1e-12


def _overconfident_logits(n: int = 4000, seed: int = 0):
    """A signal whose raw sigmoid is overconfident: true P is a squashed sigmoid
    of the logit, but the logit magnitude is inflated (temperature 3)."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 4, size=n)
    true_p = 1 / (1 + np.exp(-logits / 3.0))  # gentler true relationship
    labels = (rng.uniform(size=n) < true_p).astype(int)
    return logits, labels


def test_calibration_reduces_ece() -> None:
    logits, labels = _overconfident_logits()
    half = len(logits) // 2
    for name, factory in CALIBRATORS.items():
        cal = factory().fit(logits[:half], labels[:half])
        raw_probs = raw_sigmoid(logits[half:])
        cal_probs = cal.predict_proba(logits[half:])
        raw_ece = expected_calibration_error(raw_probs, labels[half:])
        cal_ece = expected_calibration_error(cal_probs, labels[half:])
        assert cal_ece < raw_ece, f"{name}: {cal_ece:.4f} !< {raw_ece:.4f}"


def test_group_wise_split_is_leak_free() -> None:
    # Two groups, distinct texts; the audit must report clean and no overlap.
    pool = Pool(
        texts=[f"a{i}" for i in range(10)] + [f"b{i}" for i in range(10)],
        labels=np.array([1] * 10 + [0] * 10),
        groups=["ga"] * 10 + ["gb"] * 10,
    )
    calib, test, audit = group_wise_split(pool, calib_frac=0.5, seed=0)
    assert audit["clean"]
    assert not (set(calib.groups) & set(test.groups))
    assert audit["text_overlap_count"] == 0


def test_group_wise_split_catches_cross_group_text_leak() -> None:
    # The SAME text appears in two different src groups. Whenever the split puts
    # those two groups on opposite sides, the overlap-hash audit MUST catch it.
    # (This is the negative case: proof the audit can actually fail, not just
    # report clean when clean.)
    shared = "identical essay text that leaked across two sources"
    pool = Pool(
        texts=[shared, "ga-only", shared, "gb-only"],
        labels=np.array([1, 1, 0, 0]),
        groups=["ga", "ga", "gb", "gb"],
    )
    caught = False
    for seed in range(10):
        _, _, audit = group_wise_split(pool, calib_frac=0.5, seed=seed)
        if audit["n_calib_groups"] == 1:
            # ga and gb landed on opposite sides -> the shared text crosses.
            assert not audit["clean"]
            assert audit["text_overlap_count"] >= 1
            caught = True
            break
    assert caught, "no seed separated the two groups; test could not exercise the leak"
