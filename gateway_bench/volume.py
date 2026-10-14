"""Volume and concurrency: N requests, C in flight at once, per gateway.

Answers "what happens at 100 vs 1,000 vs 10,000 requests": success rate, latency tails and the
request rate each gateway actually sustained. Closed loop (a new request starts when one finishes),
so concurrency is exact and the client never queues more than C requests.
"""
from __future__ import annotations

import asyncio
import random
from pathlib import Path

from .client import CreditGuard, chat, make_client
from .config import System
from .prompts import shape_messages
from .results import ResultWriter


async def run_volume(
    systems: list[System],
    alias: str,
    levels: list[dict],                 # [{"requests": 100, "concurrency": 10}, ...]
    shape: dict,                        # {"name", "input_tokens", "output_tokens"}
    out_path: str | Path,
    temperature: float | None = 0.0,
    seed: int = 0,
    log=print,
    progress=None,
) -> Path:
    writer = ResultWriter(Path(out_path), {"track": "V"})
    total = sum(int(lv["requests"]) for lv in levels) * len(systems)
    done = 0
    rng = random.Random(seed)
    guard = CreditGuard()
    for lv in levels:
        n, c = int(lv["requests"]), int(lv["concurrency"])
        label = f"{n}@{c}"
        order = list(systems)
        rng.shuffle(order)               # which gateway goes first changes from level to level
        for s in order:
            if guard.tripped:
                break
            log(f"  volume: {s.name} {n:,} requests, {c} at once")
            queue = list(range(n))
            async with make_client(max_connections=c + 8) as client:
                async def worker():
                    nonlocal done
                    while queue and not guard.tripped:
                        i = queue.pop()
                        r = await chat(client, s, s.model_for(alias),
                                       shape_messages(shape["input_tokens"], shape["output_tokens"], nonce=True),
                                       max_tokens=int(shape["output_tokens"]), temperature=temperature, stream=True,
                                       seed=None, meta={"level": label, "requests": n, "concurrency": c,
                                                        "shape": shape["name"], "i": i})
                        guard.record(r)
                        rec = r.to_dict()
                        rec.pop("text", None)
                        writer.write(rec)
                        done += 1
                        if progress:
                            progress(done, total)
                await asyncio.gather(*(worker() for _ in range(min(c, n))))
    writer.close()
    guard.check()
    return Path(out_path)
