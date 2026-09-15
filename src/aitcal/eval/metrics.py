"""Calibration metrics: ECE (expected calibration error) and Brier score.

ECE bins predictions by confidence and measures the average gap between mean
predicted probability and empirical accuracy per bin. Brier is the mean squared
error of the probability against the 0/1 label. We compute ECE ourselves (it is
a dozen lines and the definition matters for the headline number); Brier comes
from scikit-learn.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Equal-width-bin ECE. probs = P(positive class), labels in {0,1}."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize with the interior edges; clip so p==1.0 lands in the last bin.
    bin_idx = np.clip(np.digitize(probs, bin_edges[1:-1]), 0, n_bins - 1)

    ece = 0.0
    n = len(probs)
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        conf = float(probs[mask].mean())
        acc = float(labels[mask].mean())
        ece += (count / n) * abs(conf - acc)
    return ece


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier score. Guards against sklearn's single-class edge case."""
    labels = np.asarray(labels, dtype=int)
    return float(brier_score_loss(labels, np.asarray(probs, dtype=float)))
