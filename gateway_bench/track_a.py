"""Track A: gateway overhead with the model pinned. Systems are interleaved in random order."""
from __future__ import annotations

import asyncio
import random
import time
from pathlib import Path

from .client import chat, make_client
from .config import System, load_yaml
from .prompts import shape_messages
from .results import ResultWriter


async def run_track_a(
    systems: list[System],
    cfg_path: str | Path = "configs/track_a.yaml",
    out_dir: str | Path = "results",
    requests_per_cell: int | None = None,
    shapes_filter: list[str] | None = None,
    seed: int = 0,
    log=print,
) -> Path:
    cfg = load_yaml(cfg_path)
    alias = cfg.get("model_alias", "small")
    shapes = [s for s in cfg["shapes"] if not shapes_filter or s["name"] in shapes_filter]
    stream_modes = cfg.get("stream_modes", [True])
    n = int(requests_per_cell or cfg.get("requests_per_cell", 100))
    warmup = int(cfg.get("warmup_requests", 0))
    conc = int(cfg.get("concurrency", 8))
    nonce = bool(cfg.get("nonce", True))
    temperature = float(cfg.get("temperature", 0))

    usable = []
    for s in systems:
        if alias in s.models:
            usable.append(s)
        else:
            log(f"[skip] {s.name}: no model mapped for alias '{alias}'")
    if cfg.get("baseline") and cfg["baseline"] not in {s.name for s in usable}:
        log(f"[warn] baseline '{cfg['baseline']}' is not in this run; overhead can't be computed")

    rng = random.Random(seed)
    jobs = []
    for s in usable:
        for i in range(warmup):
            jobs.append((s, shapes[0], stream_modes[0], True, -1 - i))
    warm_jobs = jobs[:]
    jobs = []
    for rep in range(n):
        round_jobs = [(s, sh, st, False, rep) for s in usable for sh in shapes for st in stream_modes]
        rng.shuffle(round_jobs)          # randomized round-robin interleaving
        jobs.extend(round_jobs)

    out = Path(out_dir) / f"track_a_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    writer = ResultWriter(out, {"track": "A", "baseline": cfg.get("baseline"), "model_alias": alias})
    sem = asyncio.Semaphore(conc)
    done = 0
    total = len(warm_jobs) + len(jobs)

    async with make_client(max_connections=conc * 2) as client:
        async def one(job):
            nonlocal done
            s, sh, st, is_warm, rep = job
            async with sem:
                r = await chat(
                    client, s, s.model_for(alias), shape_messages(sh["input_tokens"], sh["output_tokens"], nonce=nonce),
                    max_tokens=int(sh["output_tokens"]), temperature=temperature, stream=bool(st),
                    meta={"shape": sh["name"], "warmup": is_warm, "rep": rep},
                )
            rec = r.to_dict()
            rec.pop("text", None)            # Track A doesn't need output text
            writer.write(rec)
            done += 1
            if done % 50 == 0 or done == total:
                log(f"  track A: {done}/{total}")

        # warm-up first (sequential per system is not required), then the measured jobs
        await asyncio.gather(*(one(j) for j in warm_jobs))
        await asyncio.gather(*(one(j) for j in jobs))

    writer.close()
    log(f"Track A results -> {out}")
    return out
