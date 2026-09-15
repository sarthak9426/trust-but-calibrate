"""Log-perplexity detector: the deliberately weak fairness positive-control.

Machine text is low-perplexity under a language model; human text is higher.
So a simple detector scores text by NEGATIVE log-perplexity under GPT-2:

    score = -mean_token_nll(text)     # higher = lower perplexity = more machine-like

This is exactly the mechanism the bias literature (Liang 2023) indicts: non-native
writing has HIGHER perplexity, so a perplexity detector systematically flags it as
machine. We include it so the M3 audit visibly catches a real FPR gap - a positive
control that proves the method works, not a detector we would ship.

Satisfies DetectorPort (score / score_batch -> raw logit-like score).
"""

from __future__ import annotations

GPT2_ID = "gpt2"


class LogPerplexityDetector:
    """GPT-2 negative-log-perplexity as a raw machine-likeness score."""

    name = "gpt2-log-perplexity"

    def __init__(self, model_id: str = GPT2_ID, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(model_id)
        self._model.to(self.device)
        self._model.eval()

    def _neg_mean_nll(self, text: str) -> float:
        torch = self._torch
        enc = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        input_ids = enc["input_ids"]
        if input_ids.shape[1] < 2:
            return 0.0  # too short to score; neutral
        with torch.no_grad():
            out = self._model(input_ids, labels=input_ids)
        # out.loss is the mean token NLL (cross-entropy). Lower NLL = lower
        # perplexity = more machine-like, so negate for "higher = more machine".
        return float(-out.loss.item())

    def score(self, text: str) -> float:
        return self._neg_mean_nll(text)

    def score_batch(self, texts: list[str]) -> list[float]:
        # GPT-2 perplexity is per-sequence; loop (small fairness sets, ~180 total).
        return [self._neg_mean_nll(t) for t in texts]
