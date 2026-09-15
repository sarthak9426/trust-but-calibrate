"""The one interface a detector must satisfy.

A detector maps text to a raw, uncalibrated real-valued score (a logit) where
higher means more machine-like. Owning the raw logit is the whole point: the
calibrator (M2) needs a pre-sigmoid value to apply temperature/Platt scaling.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class DetectorPort(Protocol):
    """A detector emits one raw logit per text; higher = more machine-like."""

    name: str

    def score(self, text: str) -> float:
        """Return the raw (pre-sigmoid) logit for one text."""
        ...

    def score_batch(self, texts: list[str]) -> list[float]:
        """Return raw logits for a batch of texts."""
        ...
