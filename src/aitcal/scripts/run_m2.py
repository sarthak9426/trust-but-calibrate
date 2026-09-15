"""M2 end to end: prove the RoBERTa detector is miscalibrated, then calibrate it.

  uv run python -m aitcal.scripts.run_m2 --limit 2000 --calibrator temperature

Pipeline:
  1. Load a MAGE slice (label normalized to 1=machine).
  2. Score every text with the RoBERTa detector -> raw logits.
  3. Group-wise split by `src` (leakage-audited) into calib / test.
  4. Fit the calibrator on calib logits; predict P(machine) on test.
  5. Report ECE + Brier for raw-sigmoid vs calibrated, and save the
     before/after reliability diagram.

This is the headline result: raw ECE (high) collapsing to calibrated ECE (low).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from aitcal.calibration import CALIBRATORS, raw_sigmoid
from aitcal.data import group_wise_split, load_mage
from aitcal.detectors import RobertaDetector
from aitcal.eval import brier, expected_calibration_error, reliability_diagram


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M2: calibrate RoBERTa on MAGE.")
    ap.add_argument("--limit", type=int, default=2000, help="MAGE rows to pull")
    ap.add_argument("--split", default="test", help="MAGE split to draw from")
    ap.add_argument(
        "--calibrator", default="platt", choices=list(CALIBRATORS), help="method"
    )
    ap.add_argument("--n-bins", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="docs/reliability.png", help="diagram path")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    print(f"[1/5] loading MAGE (limit={args.limit}, split={args.split})")
    pool = load_mage(split=args.split, limit=args.limit, seed=args.seed)
    frac_machine = float(np.mean(pool.labels))
    print(f"      {len(pool.texts)} rows, {frac_machine:.1%} machine, "
          f"{len(set(pool.groups))} src groups")

    print("[2/5] scoring with RoBERTa (raw logits)")
    det = RobertaDetector()
    logits = []
    bs = args.batch_size
    for i in range(0, len(pool.texts), bs):
        logits.extend(det.score_batch(pool.texts[i : i + bs]))
        print(f"      scored {min(i + bs, len(pool.texts))}/{len(pool.texts)}", end="\r")
    logits = np.array(logits, dtype=float)
    print()

    print("[3/5] group-wise split + leakage audit")
    # Split indices alongside the pool so logits follow their rows.
    calib, test, audit = group_wise_split(pool, calib_frac=0.5, seed=args.seed)
    print(f"      {audit}")
    if not audit["clean"]:
        print("      ABORT: leakage detected across the split", file=sys.stderr)
        return 2
    # Re-derive logits per subset by group membership (stable, small pools).
    # Both indices come from the split's own group sets, so a group the split
    # ever drops lands in neither subset rather than leaking into test.
    calib_groups = set(calib.groups)
    test_groups = set(test.groups)
    calib_idx = [i for i, g in enumerate(pool.groups) if g in calib_groups]
    test_idx = [i for i, g in enumerate(pool.groups) if g in test_groups]
    calib_logits, calib_y = logits[calib_idx], pool.labels[calib_idx]
    test_logits, test_y = logits[test_idx], pool.labels[test_idx]

    print(f"[4/5] fitting calibrator: {args.calibrator}")
    cal = CALIBRATORS[args.calibrator]().fit(calib_logits, calib_y)
    raw_probs = raw_sigmoid(test_logits)
    cal_probs = cal.predict_proba(test_logits)

    raw_ece = expected_calibration_error(raw_probs, test_y, args.n_bins)
    cal_ece = expected_calibration_error(cal_probs, test_y, args.n_bins)
    raw_brier, cal_brier = brier(raw_probs, test_y), brier(cal_probs, test_y)

    print("[5/5] result")
    print(f"      raw        ECE={raw_ece:.4f}  Brier={raw_brier:.4f}")
    print(f"      calibrated ECE={cal_ece:.4f}  Brier={cal_brier:.4f}")
    print(f"      ECE collapse: {raw_ece:.4f} -> {cal_ece:.4f}")

    out = reliability_diagram(
        raw_probs, cal_probs, test_y, args.out,
        n_bins=args.n_bins, raw_ece=raw_ece, cal_ece=cal_ece,
        title=f"RoBERTa on MAGE: raw vs {args.calibrator}",
    )
    print(f"      reliability diagram -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
