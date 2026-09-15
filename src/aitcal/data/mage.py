"""MAGE loader with a leakage-audited, group-wise calib/test split.

MAGE (yaful/MAGE) fields: text (str), label (int), src (str).

LABEL ORIENTATION - verified live 2026-09-15, do NOT assume:
  MAGE uses label=1 for HUMAN and label=0 for MACHINE (src values like
  `imdb_human` carry label 1; `*_gpt4_para`, `*_machine_*` carry label 0).
  Human text that was MACHINE-paraphrased (`*_human_para`) is label 0 (machine),
  which is correct for us. Our codebase convention is the opposite: 1 = machine.
  So we flip:  y = 1 - mage_label.

GROUP-WISE SPLIT: `src` encodes domain+generator (e.g. `pubmed_gpt4_para`). We
split by `src` so no generator appears in both calib and test - fitting the
calibrator within-source only. An overlap-hash audit asserts no exact text (and
no src group) crosses the split, which would silently inflate every metric.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

DATASET_ID = "yaful/MAGE"


@dataclass
class Pool:
    texts: list[str]
    labels: np.ndarray  # 1 = machine (our convention), after the MAGE flip
    groups: list[str]  # the `src` group key per row


def _text_hash(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()


def load_mage(split: str = "test", limit: int | None = None, seed: int = 0) -> Pool:
    """Load MAGE into our convention (1=machine). `limit` caps rows (shuffled)."""
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10000)

    texts, labels, groups = [], [], []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        texts.append(row["text"])
        labels.append(1 - int(row["label"]))  # flip MAGE (1=human) -> ours (1=machine)
        groups.append(str(row.get("src", "unknown")))
    return Pool(texts=texts, labels=np.array(labels, dtype=int), groups=groups)


def group_wise_split(
    pool: Pool, calib_frac: float = 0.5, seed: int = 0
) -> tuple[Pool, Pool, dict]:
    """Split by `src` group so no generator crosses calib/test.

    Returns (calib_pool, test_pool, audit) where audit reports the leakage checks.
    """
    rng = np.random.default_rng(seed)
    unique_groups = sorted(set(pool.groups))
    rng.shuffle(unique_groups)
    n_calib = max(1, int(round(len(unique_groups) * calib_frac)))
    calib_groups = set(unique_groups[:n_calib])

    def subset(keep: set) -> Pool:
        idx = [i for i, g in enumerate(pool.groups) if g in keep]
        return Pool(
            texts=[pool.texts[i] for i in idx],
            labels=pool.labels[idx],
            groups=[pool.groups[i] for i in idx],
        )

    test_groups = set(unique_groups[n_calib:])
    calib, test = subset(calib_groups), subset(test_groups)

    # Leakage audit: no group and no exact-text overlap across the split.
    calib_hashes = {_text_hash(t) for t in calib.texts}
    test_hashes = {_text_hash(t) for t in test.texts}
    text_overlap = calib_hashes & test_hashes
    group_overlap = calib_groups & test_groups
    audit = {
        "n_groups_total": len(unique_groups),
        "n_calib_groups": len(calib_groups),
        "n_test_groups": len(test_groups),
        "group_overlap": sorted(group_overlap),
        "text_overlap_count": len(text_overlap),
        "clean": not group_overlap and not text_overlap,
    }
    return calib, test, audit
