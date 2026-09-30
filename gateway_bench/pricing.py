"""Cost computation: prefer billed cost reported by the system, else list price x tokens x (1 + markup)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import load_yaml


class PriceBook:
    def __init__(self, models: dict[str, dict[str, float]], markup: dict[str, float] | None = None):
        # longest key first so "gpt-4.1-mini" beats "gpt-4.1"
        self.models = dict(sorted(models.items(), key=lambda kv: -len(kv[0])))
        self.markup = markup or {}

    @classmethod
    def load(cls, path: str | Path = "configs/prices.yaml") -> "PriceBook":
        raw = load_yaml(path)
        return cls(raw.get("models", {}) or {}, raw.get("markup", {}) or {})

    def lookup(self, model: str | None) -> dict[str, float] | None:
        if not model:
            return None
        m = model.lower()
        for key, price in self.models.items():
            if key.lower() in m:
                return price
        return None

    def list_cost(self, model: str | None, input_tokens: int | None, output_tokens: int | None) -> float | None:
        price = self.lookup(model)
        if price is None or input_tokens is None or output_tokens is None:
            return None
        return (input_tokens * price.get("input", 0.0) + output_tokens * price.get("output", 0.0)) / 1e6

    def cost(self, rec: dict[str, Any]) -> tuple[float | None, str]:
        """Return (cost_usd, source) for a result record."""
        if rec.get("cost_reported") is not None:
            return float(rec["cost_reported"]), "reported"
        model = rec.get("model_served") or rec.get("model_requested")
        base = self.list_cost(model, rec.get("input_tokens"), rec.get("output_tokens"))
        if base is None:
            return None, "unknown"
        return base * (1 + self.markup.get(rec.get("system", ""), 0.0)), "list_price"
