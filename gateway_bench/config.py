"""Load YAML configs and resolve systems."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def expand_env(value: Any) -> Any:
    """Replace ${VAR} in strings (recursively) with environment values."""
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


@dataclass
class System:
    name: str
    kind: str
    base_url: str
    api_key_env: str
    tracks: list[str] = field(default_factory=list)
    models: dict[str, str] = field(default_factory=dict)
    router_model: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    timeout_s: float = 180.0
    # How to pin the upstream provider: a body fragment where "{provider}" is replaced with the slug
    # (OpenRouter), or the string "model_prefix" to send "<provider>/<model>" as the model (Concentrate).
    pin: Any = None

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def model_for(self, alias: str) -> str:
        if alias not in self.models:
            raise KeyError(f"system '{self.name}' has no model for alias '{alias}'")
        return self.models[alias]


def load_systems(path: str | Path = "configs/systems.yaml") -> dict[str, System]:
    raw = load_yaml(path)
    defaults = raw.get("defaults", {})
    systems: dict[str, System] = {}
    for s in raw.get("systems", []):
        sys_ = System(
            name=s["name"],
            kind=s.get("kind", "gateway"),
            base_url=s["base_url"].rstrip("/"),
            api_key_env=s["api_key_env"],
            tracks=[t.upper() for t in s.get("tracks", [])],
            models=s.get("models", {}) or {},
            router_model=s.get("router_model"),
            headers=s.get("headers", {}) or {},
            extra_body=s.get("extra_body", {}) or {},
            timeout_s=float(s.get("timeout_s", defaults.get("timeout_s", 180))),
            pin=s.get("pin"),
        )
        if sys_.name in systems:
            raise ValueError(f"duplicate system name: {sys_.name}")
        systems[sys_.name] = sys_
    return systems


# Concentrate provider name -> OpenRouter provider slug. Used to pin both gateways to the same upstream.
PROVIDER_SLUGS = {
    "vertex": "google-vertex", "ai-studio": "google-ai-studio", "bedrock": "amazon-bedrock", "azure": "azure",
    "zai": "z-ai", "xai": "x-ai", "moonshot": "moonshotai", "alibaba": "alibaba", "openai": "openai",
    "anthropic": "anthropic", "deepseek": "deepseek", "deepinfra": "deepinfra", "novita": "novita",
    "fireworks": "fireworks", "baseten": "baseten", "together": "together", "cloudflare": "cloudflare",
    "crusoe": "crusoe", "wafer": "wafer", "modal": "modal", "nebius": "nebius", "runware": "runware",
    "azure-fw": "azure",
}


def concentrate_provider(or_slug: str, available: list[str] | None = None) -> str | None:
    """OpenRouter provider slug -> Concentrate's name for the same provider (None if Concentrate lacks it)."""
    cands = [c for c, o in PROVIDER_SLUGS.items() if o == or_slug] + [or_slug]
    if available is None:
        return cands[0]
    return next((c for c in cands if c in available), None)


def fill_pin(pin: Any, provider: str) -> Any:
    """Substitute {provider} into a pin template (recursively)."""
    if isinstance(pin, str):
        return pin.replace("{provider}", provider)
    if isinstance(pin, dict):
        return {k: fill_pin(v, provider) for k, v in pin.items()}
    if isinstance(pin, list):
        return [fill_pin(v, provider) for v in pin]
    return pin


def select_systems(
    systems: dict[str, System],
    track: str | None = None,
    only: list[str] | None = None,
    require_key: bool = True,
    warn=print,
) -> list[System]:
    out = []
    for s in systems.values():
        if only and s.name not in only:
            continue
        if track and track.upper() not in s.tracks:
            continue
        if require_key and not s.available:
            warn(f"[skip] {s.name}: env var {s.api_key_env} is not set")
            continue
        out.append(s)
    return out
