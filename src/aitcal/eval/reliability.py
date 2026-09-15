"""Reliability diagram: the headline visual.

Plots mean predicted probability vs empirical accuracy per bin, for the raw
(uncalibrated) detector and a calibrated one side by side, against the diagonal
(perfect calibration). This is the figure the README leads with.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _bin_stats(probs: np.ndarray, labels: np.ndarray, n_bins: int):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)
    xs, ys = [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            xs.append(float(probs[mask].mean()))
            ys.append(float(labels[mask].mean()))
    return np.array(xs), np.array(ys)


def reliability_diagram(
    raw_probs: np.ndarray,
    cal_probs: np.ndarray,
    labels: np.ndarray,
    out_path: str | Path,
    n_bins: int = 15,
    raw_ece: float | None = None,
    cal_ece: float | None = None,
    title: str = "Reliability: raw vs calibrated",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    labels = np.asarray(labels, dtype=float)
    rx, ry = _bin_stats(np.asarray(raw_probs, dtype=float), labels, n_bins)
    cx, cy = _bin_stats(np.asarray(cal_probs, dtype=float), labels, n_bins)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect")
    raw_lbl = "raw" + (f" (ECE={raw_ece:.3f})" if raw_ece is not None else "")
    cal_lbl = "calibrated" + (f" (ECE={cal_ece:.3f})" if cal_ece is not None else "")
    ax.plot(rx, ry, "o-", color="#DC2626", label=raw_lbl)
    ax.plot(cx, cy, "o-", color="#16A34A", label=cal_lbl)
    ax.set_xlabel("mean predicted P(machine)")
    ax.set_ylabel("empirical fraction machine")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.set_aspect("equal")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
