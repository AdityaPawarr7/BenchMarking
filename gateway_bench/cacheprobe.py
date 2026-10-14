"""Prompt-cache probe: does a repeated long prompt get cheaper through each gateway?

Agents and chat apps resend the same long prefix (system prompt, tools, history) on every call.
Providers bill cached input at a fraction of the normal price, but only if the gateway sends repeats
to the same provider and passes the discount through. This probe sends one long prompt, unique to
this run, several times in a row per gateway and records cached input tokens and billed cost.
"""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from .client import chat, make_client
from .config import System
from .prompts import filler_text
from .results import ResultWriter


def cache_messages(prefix_tokens: int, tag: str) -> list[dict]:
    import random
    body = filler_text(prefix_tokens, random.Random(tag))
    return [{"role": "system", "content": f"[probe {tag}] You are a benchmarking assistant.\n\n{body}"},
            {"role": "user", "content": "Reply with the single word OK."}]


async def run_cache_probe(
    systems: list[System],
    alias: str,
    out_path: str | Path,
    prefix_tokens: int = 8000,
    repeats: int = 4,
    gap_s: float = 2.0,
    temperature: float | None = 0.0,
    log=print,
    progress=None,
) -> Path:
    writer = ResultWriter(Path(out_path), {"track": "C"})
    total, done = len(systems) * repeats, 0
    async with make_client(max_connections=4) as client:
        for s in systems:
            tag = uuid.uuid4().hex[:10]               # new prefix per gateway: the first call is always a miss
            msgs = cache_messages(prefix_tokens, tag)
            log(f"  cache probe: {s.name}, {repeats} identical ~{prefix_tokens:,}-token prompts")
            for k in range(repeats):
                r = await chat(client, s, s.model_for(alias), msgs, max_tokens=8, temperature=temperature,
                               stream=False, seed=None, meta={"repeat": k, "prefix_tokens": prefix_tokens})
                rec = r.to_dict()
                rec.pop("text", None)
                writer.write(rec)
                done += 1
                if progress:
                    progress(done, total)
                await asyncio.sleep(gap_s)
    writer.close()
    return Path(out_path)
