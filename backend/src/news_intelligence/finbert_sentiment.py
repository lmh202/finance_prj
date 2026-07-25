"""FinBERT sentiment scoring — Developer 3 (live pipeline).

Replaces the old keyword-count sentiment heuristic in analyzer.py with
ProsusAI/finbert, a BERT model fine-tuned on financial text for 3-class
sentiment (positive/negative/neutral). The model loads lazily and once per
process — the first score() call pays the load cost, every call after
reuses the cached model/tokenizer.

Weights are persisted to data/models/finbert/ (under DATA_DIR, so
AURORA_DATA_DIR sandboxes get their own copy) the first time they're
downloaded, and every load after that reads ONLY from that local copy
(local_files_only=True — no Hub network call at all, not even a
reachability check) so the live pipeline keeps working with no network
access once the weights exist on disk.

analyzer.py still falls back to the keyword scorer if this raises
ModelUnavailable (transformers/torch missing, or — on a machine that has
never downloaded the weights — no network to fetch them) — the same
"never breaks the demo" pattern this engine already uses for the LLM
classification path.
"""

from __future__ import annotations

from typing import Dict, Tuple

from src.config import DATA_DIR

MODEL_NAME = "ProsusAI/finbert"
WEIGHTS_DIR = DATA_DIR / "models" / "finbert"

_model = None
_tokenizer = None
_device = None


class ModelUnavailable(Exception):
    """transformers/torch aren't installed, or the model failed to load."""


def _load() -> None:
    global _model, _tokenizer, _device
    if _model is not None:
        return

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ModelUnavailable(f"transformers/torch not installed: {exc}") from exc

    # A prior successful load leaves a full HF save_pretrained() dump here
    # (config.json + weights + tokenizer files) — its presence means we can
    # load without touching the network at all. Otherwise this is the
    # first-ever run: fetch from the Hub, then persist that same dump here
    # so every later run (including offline ones) takes the cached branch.
    cached = (WEIGHTS_DIR / "config.json").exists()
    source = WEIGHTS_DIR if cached else MODEL_NAME
    try:
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=cached)
        model = AutoModelForSequenceClassification.from_pretrained(source, local_files_only=cached)
    except Exception as exc:  # any download/load failure -> caller falls back
        raise ModelUnavailable(f"could not load {MODEL_NAME}: {exc}") from exc

    if not cached:
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(WEIGHTS_DIR)
        model.save_pretrained(WEIGHTS_DIR)

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    _model, _tokenizer, _device = model, tokenizer, device


def score(text: str) -> Tuple[float, str]:
    """(-1..1 signed score, 'positive'|'negative'|'neutral') for `text`.

    Score is P(positive) - P(negative) from FinBERT's 3-class softmax
    (naturally in [-1, 1]); label is FinBERT's own argmax class rather than
    a threshold on the scalar, since the scalar collapses information the
    label shouldn't lose (e.g. positive-vs-neutral is not a sign flip).

    Raises ModelUnavailable if the model can't be loaded — callers should
    catch this and fall back to a keyword heuristic.
    """
    _load()
    import torch

    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = _model(**inputs).logits[0]
    probs = torch.softmax(logits, dim=-1)

    by_label: Dict[str, float] = {
        _model.config.id2label[i].lower(): probs[i].item() for i in range(len(probs))
    }
    label = max(by_label, key=by_label.get)
    signed = by_label.get("positive", 0.0) - by_label.get("negative", 0.0)
    return round(signed, 3), label
