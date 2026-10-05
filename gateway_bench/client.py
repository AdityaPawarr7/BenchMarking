"""Async OpenAI-compatible client that records timing, tokens, served model and cost."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from .config import System, expand_env


@dataclass
class CallResult:
    request_id: str
    system: str
    model_requested: str
    model_served: str | None = None
    stream: bool = True
    status: int | None = None
    ok: bool = False
    error: str | None = None
    t_start: float = 0.0          # unix time (s) the request was sent
    ttft_ms: float | None = None  # time to first token of any kind incl. reasoning (stream) or full response (non-stream)
    first_text_ms: float | None = None  # time to first visible answer text (stream)
    e2e_ms: float | None = None   # time to last byte
    input_tokens: int | None = None
    output_tokens: int | None = None
    output_tps: float | None = None
    cost_reported: float | None = None   # billed cost if the system returns it (e.g. usage.cost)
    text: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_headers(system: System) -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if system.api_key:
        h["Authorization"] = f"Bearer {system.api_key}"
    h.update(expand_env(system.headers))
    return h


def _extract_cost(usage: dict[str, Any] | None) -> float | None:
    if not usage:
        return None
    for key in ("cost", "total_cost", "cost_usd"):
        v = usage.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    return None


async def chat(
    client: httpx.AsyncClient,
    system: System,
    model: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = 512,
    temperature: float | None = 0.0,
    stream: bool = True,
    seed: int | None = 1234,
    extra_body: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> CallResult:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if temperature is not None:
        body["temperature"] = temperature
    if seed is not None:
        body["seed"] = seed
    if stream:
        body["stream_options"] = {"include_usage": True}
    body.update(expand_env(system.extra_body))
    if extra_body:
        body.update(extra_body)

    res = CallResult(
        request_id=str(uuid.uuid4()),
        system=system.name,
        model_requested=model,
        stream=stream,
        meta=meta or {},
    )
    url = f"{system.base_url}/chat/completions"
    headers = build_headers(system)
    res.t_start = time.time()
    t0 = time.perf_counter()
    try:
        # httpx's timeout is per read, so a connection that keeps trickling bytes never times out.
        # wait_for caps the whole request.
        coro = (_do_stream if stream else _do_plain)(client, url, headers, body, res, t0, system.timeout_s)
        await asyncio.wait_for(coro, timeout=system.timeout_s)
    except asyncio.TimeoutError:
        res.ok = False
        res.error = f"timeout: no complete response within {system.timeout_s:.0f}s"
    except httpx.TimeoutException as e:
        res.error = f"timeout: {e!r}"
    except httpx.HTTPError as e:
        res.error = f"http_error: {e!r}"
    except Exception as e:  # noqa: BLE001 - benchmarking must never crash on one call
        res.error = f"exception: {e!r}"

    if res.e2e_ms is None:
        res.e2e_ms = (time.perf_counter() - t0) * 1000
    if res.ok and res.output_tokens and res.ttft_ms is not None and stream:
        gen_s = (res.e2e_ms - res.ttft_ms) / 1000
        if gen_s > 0:
            res.output_tps = res.output_tokens / gen_s
    return res


async def _do_plain(client, url, headers, body, res: CallResult, t0: float, timeout: float) -> None:
    r = await client.post(url, headers=headers, json=body, timeout=timeout)
    res.e2e_ms = (time.perf_counter() - t0) * 1000
    res.ttft_ms = res.e2e_ms
    res.status = r.status_code
    if r.status_code >= 400:
        res.error = f"status {r.status_code}: {r.text[:4000]}"
        return
    data = r.json()
    res.model_served = data.get("model")
    choices = data.get("choices") or []
    if choices:
        res.text = (choices[0].get("message") or {}).get("content") or ""
    usage = data.get("usage") or {}
    res.input_tokens = usage.get("prompt_tokens")
    res.output_tokens = usage.get("completion_tokens")
    res.cost_reported = _extract_cost(usage)
    res.meta["gateway_headers"] = _interesting_headers(r.headers)
    res.ok = True


async def _do_stream(client, url, headers, body, res: CallResult, t0: float, timeout: float) -> None:
    parts: list[str] = []
    async with client.stream("POST", url, headers=headers, json=body, timeout=timeout) as r:
        res.status = r.status_code
        res.meta["gateway_headers"] = _interesting_headers(r.headers)
        if r.status_code >= 400:
            txt = (await r.aread()).decode("utf-8", "replace")
            res.error = f"status {r.status_code}: {txt[:4000]}"
            return
        async for line in r.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if chunk.get("error"):
                res.error = f"stream_error: {json.dumps(chunk['error'])[:500]}"
                return
            if chunk.get("model") and not res.model_served:
                res.model_served = chunk["model"]
            for ch in chunk.get("choices") or []:
                d = ch.get("delta") or {}
                text = d.get("content")
                thinking = d.get("reasoning") or d.get("reasoning_content") or d.get("thinking") \
                    or d.get("reasoning_details")
                if (text or thinking) and res.ttft_ms is None:
                    res.ttft_ms = (time.perf_counter() - t0) * 1000
                if text:
                    if res.first_text_ms is None:
                        res.first_text_ms = (time.perf_counter() - t0) * 1000
                    parts.append(text)
            usage = chunk.get("usage")
            if usage:
                res.input_tokens = usage.get("prompt_tokens")
                res.output_tokens = usage.get("completion_tokens")
                res.cost_reported = _extract_cost(usage)
    res.e2e_ms = (time.perf_counter() - t0) * 1000
    res.text = "".join(parts)
    res.ok = True


_HEADER_KEYS = ("x-request-id", "x-ratelimit", "x-cache", "cf-aig", "x-portkey", "x-helicone",
                "x-litellm", "x-openrouter", "x-concentrate", "x-ramp", "x-served-by", "x-model")


def _interesting_headers(headers: httpx.Headers) -> dict[str, str]:
    return {k: v for k, v in headers.items() if any(k.lower().startswith(p) for p in _HEADER_KEYS)}


def make_client(max_connections: int = 256) -> httpx.AsyncClient:
    limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections)
    return httpx.AsyncClient(limits=limits, http2=False)


class OutOfCredits(RuntimeError):
    pass


class CreditGuard:
    """Stops a run when a gateway keeps answering 402 (out of credits): the remaining requests
    would only fail, and the other gateway's requests would be wasted money."""

    def __init__(self, limit: int = 5):
        self.limit, self.streak, self.tripped = limit, {}, None

    def record(self, r: "CallResult") -> None:
        # Count every 402, not just consecutive ones: when credits run low, cheap requests still pass
        # while expensive ones fail, so failures interleave with successes and skew the sample.
        if r.status == 402:
            self.streak[r.system] = self.streak.get(r.system, 0) + 1
            if self.streak[r.system] >= self.limit and not self.tripped:
                self.tripped = (f"{r.system} returned 402 (out of credits) {self.limit} times; stopped the run so "
                                f"the comparison isn't skewed and no more money is spent. Top up {r.system} and "
                                f"run again. Last error: {(r.error or '')[:200]}")

    def check(self) -> None:
        if self.tripped:
            raise OutOfCredits(self.tripped)
