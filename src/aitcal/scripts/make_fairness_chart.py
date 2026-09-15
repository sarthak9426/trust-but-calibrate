"""Render the fairness bar chart across both detectors -> docs/fairness.png.

  uv run python -m aitcal.scripts.make_fairness_chart

Runs the native-anchored FPR audit for each detector and draws grouped bars
(native vs non-native FPR) with a bootstrap-CI whisker on the non-native bar.
"""

from __future__ import annotations

import argparse

from aitcal.data.toefl import load_fairness_data
from aitcal.eval import fairness_bar_chart
from aitcal.scripts.run_m3 import DETECTORS, audit_detector

# Human-readable bar labels.
LABELS = {"perplexity": "GPT-2 log-perplexity\n(weak control)", "roberta": "RoBERTa"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render the fairness bar chart.")
    ap.add_argument("--out", default="docs/fairness.png")
    ap.add_argument("--target-native-fpr", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    data = load_fairness_data()
    results = {}
    for name in DETECTORS:
        print(f"auditing {name} ...")
        res = audit_detector(
            name, target_native_fpr=args.target_native_fpr, seed=args.seed, data=data
        )
        results[LABELS.get(name, name)] = res
        print(f"  native {res['fpr_native']:.0%}  non-native {res['fpr_nonnative']:.0%}  "
              f"gap {res['gap']:+.0%} [{res['ci_low']:+.0%}, {res['ci_high']:+.0%}]")

    out = fairness_bar_chart(results, args.out, native_target=args.target_native_fpr)
    print(f"fairness chart -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
