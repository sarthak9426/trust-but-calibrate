"""Liang 2023 non-native fairness data (ChatGPT-Detector-Bias repo).

Two groups of HUMAN essays:
  - non-native: 91 real TOEFL essays by non-native English writers
  - native:     88 real US student essays (Hewlett set)

Source: github.com/Weixin-Liang/ChatGPT-Detector-Bias, Data_and_Results/Human_Data.
Each data.json is a list of {"document": <essay text>}. Both groups are human,
so the only detector error possible is a false positive.

Files are fetched once and cached under a local dir (git-ignored).
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_RAW = (
    "https://raw.githubusercontent.com/Weixin-Liang/ChatGPT-Detector-Bias/main/"
    "Data_and_Results/Human_Data"
)
NONNATIVE_PATH = "TOEFL_real_91/data.json"
NATIVE_PATH = "HewlettStudentEssay_real_88/data.json"

CACHE_DIR = Path("data/toefl")


@dataclass
class FairnessData:
    nonnative: list[str]  # non-native (TOEFL) human essays
    native: list[str]  # native (Hewlett) human essays


def _fetch(rel_path: str) -> list[str]:
    cache = CACHE_DIR / rel_path.replace("/", "_")
    if cache.exists():
        raw = cache.read_text(encoding="utf-8")
    else:
        raw = urllib.request.urlopen(f"{_RAW}/{rel_path}").read().decode("utf-8")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(raw, encoding="utf-8")
    data = json.loads(raw)
    # Each element is {"document": "..."}.
    return [row["document"] for row in data if row.get("document")]


def load_fairness_data() -> FairnessData:
    return FairnessData(
        nonnative=_fetch(NONNATIVE_PATH),
        native=_fetch(NATIVE_PATH),
    )
