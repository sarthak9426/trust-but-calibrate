"""A tiny in-repo labeled sample for the M1 smoke test.

Not a benchmark - just enough clearly-human and clearly-machine text to confirm
the detector loads and emits sane raw logits (machine text should score higher).
The real calibration/fairness pools (MAGE, TOEFL) arrive in M2/M3.
"""

from __future__ import annotations

# label 1 = machine-generated, label 0 = human-written.
SAMPLE: list[tuple[str, int]] = [
    (
        "I went to the store yesterday and honestly the line was insane, "
        "so I just grabbed milk and bailed. Cashier was chatty though.",
        0,
    ),
    (
        "my kid drew on the wall again with a sharpie lol. spent an hour "
        "scrubbing it and it's still faintly there. parenting is a scam.",
        0,
    ),
    (
        "Furthermore, it is important to note that the implementation of "
        "sustainable practices can significantly enhance operational "
        "efficiency while simultaneously reducing environmental impact across "
        "multiple organizational dimensions.",
        1,
    ),
    (
        "In conclusion, the aforementioned considerations underscore the "
        "multifaceted nature of the challenge, necessitating a comprehensive "
        "and holistic approach to effectively address the underlying issues.",
        1,
    ),
]


def texts_and_labels() -> tuple[list[str], list[int]]:
    texts = [t for t, _ in SAMPLE]
    labels = [y for _, y in SAMPLE]
    return texts, labels
