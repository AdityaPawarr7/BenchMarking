"""Open-loop load test: Poisson arrivals at stepped target rates."""
from __future__ import annotations

import asyncio
import random
import time
from pathlib import Path

from .client import chat, make_client
from .config import System, load_yaml
from .prompts import shape_messages
from .results import ResultWriter


async def run_load(
    systems: list[System],
    cfg_path: str | Path = "configs/load.yaml",
    out_dir: str | Path = "results",
    rates: list[float] | None = None,
    duration_s: float | None = None,
    seed: int = 0,
    log=print,
) -> Path:
    cfg = load_yaml(cfg_path)
    alias = cfg.get("model_alias", "small")
    shape = cfg["shape"]
    rates = rates or cfg["rates_rps"]
    duration = float(duration_s or cfg.get("step_duration_s", 60))
    stream = bool(cfg.get("stream", True))
    out = Path(out_dir) / f"load_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    writer = ResultWriter(out, {"track": "load", "model_alias": alias})
    rng = random.Random(seed)

    async with make_client(max_connections=2048) as client:
        for s in systems:
            if alias not in s.models:
                log(f"[skip] {s.name}: no model for alias '{alias}'")
                continue
            for rate in rates:
                log(f"  load: {s.name} @ {rate} rps for {duration:.0f}s")
                tasks = []
                t_end = time.perf_counter() + duration
                while time.perf_counter() < t_end:
                    msgs = shape_messages(shape["input_tokens"], shape["output_tokens"], nonce=True)
                    tasks.append(asyncio.create_task(_one(client, s, alias, msgs, shape, stream, rate, writer)))
                    await asyncio.sleep(rng.expovariate(rate))   # open loop: don't wait for responses
                await asyncio.gather(*tasks)
    writer.close()
    log(f"Load results -> {out}")
    return out


async def _one(client, s, alias, msgs, shape, stream, rate, writer):
    r = await chat(client, s, s.model_for(alias), msgs, max_tokens=int(shape["output_tokens"]),
                   stream=stream, meta={"target_rps": rate})
    rec = r.to_dict()
    rec.pop("text", None)
    writer.write(rec)
