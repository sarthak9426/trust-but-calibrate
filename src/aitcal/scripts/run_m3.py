"""M3 end to end: the native-vs-non-native false-positive fairness audit.

  uv run python -m aitcal.scripts.run_m3 --detector perplexity --threshold 0.5

Every essay in the fairness set is HUMAN, so a detector flagging one as machine
is a false positive. We measure FPR for non-native (TOEFL) vs native (Hewlett)
writers and report the gap with a bootstrap CI.

Detector choice:
  - roberta:    the M1/M2 detector (modern-ish, weak by 2026 standards)
  - perplexity: GPT-2 log-perplexity, the deliberately weak positive control the
                bias literature indicts - expected to show the largest gap.

Thresholding: a raw score needs a cut to become a machine/human call. The
default `native-fpr` mode tunes the cut so the NATIVE group is flagged at a
low target rate (10%) - the operating point a fair deployment would pick - then
measures how often NON-NATIVE writers are flagged at that same cut. `median`
(combined-median) and `fixed` modes are available but inflate the absolute FPRs;
the gap is the signal, and native-anchoring is the defensible headline.
"""

from __future__ import annotations

import argparse

import numpy as np

from aitcal.data.toefl import load_fairness_data
from aitcal.eval.fairness import fpr_gap_bootstrap

DETECTORS = ("roberta", "perplexity")


def _build_detector(name: str):
    if name == "roberta":
        from aitcal.detectors import RobertaDetector

        return RobertaDetector()
    if name == "perplexity":
        from aitcal.detectors.perplexity import LogPerplexityDetector

        return LogPerplexityDetector()
    raise ValueError(f"unknown detector {name!r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M3: native-vs-non-native FPR audit.")
    ap.add_argument("--detector", default="perplexity", choices=DETECTORS)
    ap.add_argument(
        "--threshold-mode",
        default="native-fpr",
        choices=("native-fpr", "median", "fixed"),
        help="how to set the machine/human cut (default: anchor to a target native FPR)",
    )
    ap.add_argument(
        "--target-native-fpr",
        type=float,
        default=0.10,
        help="native FPR to tune the cut to, in native-fpr mode",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="raw-score cut, in fixed mode",
    )
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    print(f"[1/3] loading fairness data (Liang 2023)")
    data = load_fairness_data()
    print(f"      non-native (TOEFL): {len(data.nonnative)}  "
          f"native (Hewlett): {len(data.native)}")

    print(f"[2/3] scoring with '{args.detector}' (raw machine-likeness score)")
    det = _build_detector(args.detector)
    nn_scores = np.array(det.score_batch(data.nonnative), dtype=float)
    na_scores = np.array(det.score_batch(data.native), dtype=float)

    # Pick the operating point.
    if args.threshold_mode == "fixed":
        if args.threshold is None:
            ap.error("--threshold is required in fixed mode")
        thresh = args.threshold
        thresh_desc = f"fixed raw score {thresh:.3f}"
    elif args.threshold_mode == "median":
        thresh = float(np.median(np.concatenate([nn_scores, na_scores])))
        thresh_desc = f"combined median {thresh:.3f}"
    else:  # native-fpr: tune the cut so the NATIVE group's FPR hits the target.
        # Native FPR = fraction of native scores above the cut. To get a target
        # native FPR t, set the cut at the (1 - t) quantile of native scores.
        q = 1.0 - args.target_native_fpr
        thresh = float(np.quantile(na_scores, q))
        thresh_desc = (
            f"native-anchored (target native FPR {args.target_native_fpr:.0%}, "
            f"cut {thresh:.3f})"
        )

    nn_pred = (nn_scores > thresh).astype(int)  # 1 = flagged machine (false positive)
    na_pred = (na_scores > thresh).astype(int)

    print(f"[3/3] FPR gap + bootstrap CI (threshold: {thresh_desc})")
    res = fpr_gap_bootstrap(nn_pred, na_pred, n_boot=args.n_boot, seed=args.seed)
    print(f"      FPR non-native = {res['fpr_nonnative']:.1%}  (n={res['n_nonnative']})")
    print(f"      FPR native     = {res['fpr_native']:.1%}  (n={res['n_native']})")
    print(f"      gap            = {res['gap']:+.1%}  "
          f"[{res['ci_level']:.0%} CI {res['ci_low']:+.1%}, {res['ci_high']:+.1%}]")
    verdict = (
        "gap supported (CI excludes 0)"
        if res["excludes_zero"]
        else "gap NOT distinguishable from 0 at this n"
    )
    print(f"      verdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
