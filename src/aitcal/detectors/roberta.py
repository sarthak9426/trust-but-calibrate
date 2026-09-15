"""RoBERTa adapter: openai-community/roberta-base-openai-detector.

The model is a 2-class sequence classifier. Note the label order is
{0: 'Fake', 1: 'Real'} on this checkpoint, so we read the indices from the
model config rather than hardcoding them - a different checkpoint may order
them the other way. We expose the machine-vs-human logit difference
(logit[fake] - logit[real]) as the detector score, so higher = more
machine-like and the value is a genuine pre-sigmoid logit the calibrator owns.
"""

from __future__ import annotations

MODEL_ID = "openai-community/roberta-base-openai-detector"


class RobertaDetector:
    """Wraps the openai-community RoBERTa detector; emits a raw logit."""

    name = "roberta-base-openai-detector"

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        # Import torch/transformers lazily so importing the port graph (and the
        # rest of the harness) does not require the heavy ML stack.
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self._model.to(self.device)
        self._model.eval()

        # Resolve which logit index is the machine/human class from the config,
        # matching label text case-insensitively. A different checkpoint may
        # order Fake/Real the other way, so never hardcode the index.
        id2label = {i: str(v).lower() for i, v in self._model.config.id2label.items()}
        self._fake_idx = self._find_index(id2label, ("fake", "machine", "generated", "ai"))
        self._real_idx = self._find_index(id2label, ("real", "human"))

    @staticmethod
    def _find_index(id2label: dict[int, str], keywords: tuple[str, ...]) -> int:
        for idx, label in id2label.items():
            if any(k in label for k in keywords):
                return idx
        raise ValueError(f"no label matching {keywords} in {id2label}")

    def score(self, text: str) -> float:
        return self.score_batch([text])[0]

    def score_batch(self, texts: list[str]) -> list[float]:
        torch = self._torch
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(self.device)
        with torch.no_grad():
            logits = self._model(**inputs).logits  # shape (N, 2)
        # Machine-vs-human logit difference is the raw score we calibrate.
        score = logits[:, self._fake_idx] - logits[:, self._real_idx]
        return score.cpu().tolist()
