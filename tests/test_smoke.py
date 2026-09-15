"""M1 smoke test: the RoBERTa detector loads and emits sane raw logits.

Marked `slow` because it downloads the model on first run. Run explicitly with
`uv run pytest -m slow -s` to see the printed logits, or the default
`uv run pytest` which still executes it (there is no deselect by default).
"""

from __future__ import annotations

import pytest

from aitcal.data import texts_and_labels
from aitcal.detectors import DetectorPort, RobertaDetector


@pytest.fixture(scope="module")
def detector() -> RobertaDetector:
    return RobertaDetector()


def test_adapter_satisfies_port(detector: RobertaDetector) -> None:
    # The adapter is structurally a DetectorPort.
    assert isinstance(detector, DetectorPort)


def test_emits_raw_logits(detector: RobertaDetector) -> None:
    texts, labels = texts_and_labels()
    scores = detector.score_batch(texts)

    # Plumbing invariants (these are the M1 gate): one finite float per text.
    assert len(scores) == len(texts)
    assert all(isinstance(s, float) for s in scores)
    import math

    assert all(math.isfinite(s) for s in scores)

    # Print so `-s` shows the raw logits (the M1 deliverable).
    print("\n--- raw logits (higher = more machine-like) ---")
    for text, label, score in zip(texts, labels, scores):
        tag = "MACHINE" if label == 1 else "human  "
        print(f"[{tag}] logit={score:+.3f}  {text[:60]!r}")

    # Ranking is a DIAGNOSTIC, not a gate: this detector is old and weak (that
    # is why it doubles as the M3 fairness positive-control), and the toy
    # samples are stylized, so a wrong ranking here reflects the detector, not a
    # wiring bug. We report it rather than assert on it.
    machine = [s for s, y in zip(scores, labels) if y == 1]
    human = [s for s, y in zip(scores, labels) if y == 0]
    mean_m, mean_h = sum(machine) / len(machine), sum(human) / len(human)
    print(f"mean machine logit={mean_m:+.3f}  mean human logit={mean_h:+.3f}  "
          f"(separates correctly: {mean_m > mean_h})")


def test_single_score_matches_batch(detector: RobertaDetector) -> None:
    texts, _ = texts_and_labels()
    single = detector.score(texts[0])
    batched = detector.score_batch([texts[0]])[0]
    assert single == pytest.approx(batched, abs=1e-4)
