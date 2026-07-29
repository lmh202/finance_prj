"""DeepSeek explanation client for already-fixed numeric decisions.

The LLM is intentionally outside the numeric path.  It can summarize the
optimizer's output, but it cannot add, remove, resize, or reverse trades.
When credentials, the network, or validation are unavailable, callers receive
a deterministic template generated from the same payload.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Dict, List

import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
REQUEST_TIMEOUT_SECONDS = 20
MAX_CACHE_ENTRIES = 256

_CACHE: Dict[str, dict] = {}
_CACHE_LOCK = threading.Lock()


class DecisionExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=600)
    reasons: List[str] = Field(default_factory=list, max_length=4)
    cautions: List[str] = Field(default_factory=list, max_length=3)
    confidence_note: str = Field(default="", max_length=300)


def _cache_key(payload: dict, model: str) -> str:
    encoded = json.dumps(
        {"model": model, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _trade_sentence(trade: dict) -> str:
    change = float(trade.get("weight_change_pct", 0.0))
    direction = "increase" if change > 0 else "reduce"
    return (
        f"{direction} {trade.get('symbol', 'the position')} by "
        f"{abs(change):.1f} percentage points"
    )


def deterministic_explanation(
    payload: dict,
    reason: str = "template",
) -> dict:
    trades = list(payload.get("trades") or [])
    mode = str(payload.get("production_mode") or "risk_only")
    if trades:
        action_text = "; ".join(_trade_sentence(trade) for trade in trades)
        summary = f"Suggested adjustment: {action_text}."
    else:
        summary = (
            "No position change clears the return, risk, cost, and portfolio "
            "constraints today."
        )
    if mode == "risk_only":
        reasons = [
            "The production decision is based on the calibrated near-term risk forecast.",
            "The return model has not passed every promotion gate.",
        ]
    else:
        reasons = [
            "Expected relative return and HAR-X + News risk agree on the adjustment.",
            "Transaction costs and turnover are included before the trade is shown.",
        ]
    return {
        "summary": summary,
        "reasons": reasons,
        "cautions": [
            "This is decision support, not automatic trade execution."
        ],
        "confidence_note": str(payload.get("confidence_note") or ""),
        "_meta": {
            "source": "deterministic_template",
            "reason": reason,
            "model": None,
        },
    }


def explain_decision(payload: dict) -> dict:
    """Return a validated explanation without changing numeric decisions."""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        return deterministic_explanation(payload, "missing_api_key")

    base_url = (
        os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)
        .strip()
        .rstrip("/")
    )
    model = os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL).strip()
    key = _cache_key(payload, model)
    with _CACHE_LOCK:
        if key in _CACHE:
            cached = dict(_CACHE[key])
            cached["_meta"] = dict(cached["_meta"])
            cached["_meta"]["cache_hit"] = True
            return cached

    system_prompt = """You explain an investment decision that has already
been fixed by a deterministic optimizer. You must not add, remove, resize, or
reverse any trade. Do not predict price direction. Explain the supplied
return, volatility, cost, and constraint evidence in plain language.
Return JSON only with exactly these fields:
{
  "summary": "one concise paragraph",
  "reasons": ["up to four concise reasons"],
  "cautions": ["up to three concise cautions"],
  "confidence_note": "one concise sentence"
}
"""
    request_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Explain this fixed decision as JSON. Numeric actions are "
                    "immutable:\n"
                    + json.dumps(payload, default=str, separators=(",", ":"))
                ),
            },
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
        "max_tokens": 700,
        "stream": False,
    }
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=request_payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        parsed = DecisionExplanation.model_validate(json.loads(content))
        result = parsed.model_dump()
        result["_meta"] = {
            "source": "deepseek",
            "reason": "ok",
            "model": body.get("model", model),
            "request_id": body.get("id"),
            "cache_hit": False,
        }
    except (
        requests.RequestException,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        ValidationError,
    ) as exc:
        return deterministic_explanation(
            payload,
            f"deepseek_error:{type(exc).__name__}",
        )

    with _CACHE_LOCK:
        if len(_CACHE) >= MAX_CACHE_ENTRIES:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = result
    return dict(result)
