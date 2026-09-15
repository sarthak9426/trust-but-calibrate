"""Calibrators: map a detector's raw logit to a calibrated P(machine).

Three off-the-shelf methods, one interface (fit on held-out calib logits, then
predict_proba on test logits):

- temperature: divide the logit by a single learned scalar T, then sigmoid. The
  standard fix for an overconfident classifier; preserves ranking, one parameter.
- platt: logistic regression on the raw logit (learns both scale and shift).
- isotonic: non-parametric monotone fit; most flexible, needs more calib data.

We do not reimplement the math: temperature scaling minimizes NLL with scipy,
Platt/isotonic come from scikit-learn.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class TemperatureScaler:
    """Single-scalar temperature scaling: P = sigmoid(logit / T)."""

    def __init__(self) -> None:
        self.temperature: float = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        logits = np.asarray(logits, dtype=float)
        labels = np.asarray(labels, dtype=float)

        def nll(t: float) -> float:
            t = max(t, 1e-3)
            p = np.clip(expit(logits / t), 1e-7, 1 - 1e-7)
            return float(-(labels * np.log(p) + (1 - labels) * np.log(1 - p)).mean())

        res = minimize_scalar(nll, bounds=(1e-2, 100.0), method="bounded")
        self.temperature = float(res.x)
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        return expit(np.asarray(logits, dtype=float) / self.temperature)


class PlattScaler:
    """Logistic regression on the raw logit (learns scale + shift)."""

    def __init__(self) -> None:
        self._lr = LogisticRegression(C=1e6)  # near-unregularized: a 1-D fit

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "PlattScaler":
        self._lr.fit(np.asarray(logits, dtype=float).reshape(-1, 1), np.asarray(labels))
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        x = np.asarray(logits, dtype=float).reshape(-1, 1)
        return self._lr.predict_proba(x)[:, 1]


class IsotonicScaler:
    """Non-parametric monotone calibration."""

    def __init__(self) -> None:
        self._iso = IsotonicRegression(out_of_bounds="clip")

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "IsotonicScaler":
        self._iso.fit(np.asarray(logits, dtype=float), np.asarray(labels, dtype=float))
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        return self._iso.predict(np.asarray(logits, dtype=float))


def raw_sigmoid(logits: np.ndarray) -> np.ndarray:
    """The uncalibrated baseline: sigmoid of the raw logit, no scaling."""
    return expit(np.asarray(logits, dtype=float))


CALIBRATORS = {
    "temperature": TemperatureScaler,
    "platt": PlattScaler,
    "isotonic": IsotonicScaler,
}
