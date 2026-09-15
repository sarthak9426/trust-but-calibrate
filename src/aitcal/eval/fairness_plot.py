"""Fairness bar chart: native vs non-native FPR per detector.

For each detector, two bars (native, non-native FPR) at the native-anchored
operating point, with a whisker on the non-native bar showing the bootstrap CI
on the GAP placed at native + [gap_lo, gap_hi] (so the whisker reads as "how far
above native, with uncertainty"). A dashed line marks the native target FPR.
"""

from __future__ import annotations

from pathlib import Path


def fairness_bar_chart(
    results: dict[str, dict],
    out_path: str | Path,
    native_target: float = 0.10,
    title: str = "False-positive rate on human text: native vs non-native",
) -> Path:
    """results: {detector_name: fpr_gap_bootstrap(...) dict}."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(results)
    x = np.arange(len(names))
    width = 0.38

    native = [results[n]["fpr_native"] for n in names]
    nonnative = [results[n]["fpr_nonnative"] for n in names]
    # CI on the gap, expressed as absolute non-native positions around the point.
    err_low = [results[n]["fpr_nonnative"] - (results[n]["fpr_native"] + results[n]["ci_low"])
               for n in names]
    err_high = [(results[n]["fpr_native"] + results[n]["ci_high"]) - results[n]["fpr_nonnative"]
                for n in names]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, native, width, label="native", color="#0284C7")
    ax.bar(
        x + width / 2, nonnative, width, label="non-native", color="#DC2626",
        yerr=[np.abs(err_low), np.abs(err_high)], capsize=6, ecolor="#7F1D1D",
    )
    ax.axhline(native_target, ls="--", lw=1, color="#666",
               label=f"native target {native_target:.0%}")

    for xi, (na, nn) in enumerate(zip(native, nonnative)):
        ax.text(xi - width / 2, na + 0.01, f"{na:.0%}", ha="center", fontsize=9)
        ax.text(xi + width / 2, nn + 0.01, f"{nn:.0%}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("false-positive rate (human flagged as machine)")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper right")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
