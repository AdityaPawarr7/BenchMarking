"""Quality parity: the same model, the same questions, through each gateway.

A gateway should not change what a model knows. If the same pinned model scores lower through one
gateway, something in between is different (a cheaper quantization, a different provider, dropped
parameters, truncated output). Each question goes to every gateway, so results are paired and can be
tested with McNemar's test.
"""
from __future__ import annotations

import asyncio
import random
from pathlib import Path
from typing import Any

from .client import CreditGuard, chat, make_client
from .config import System, load_yaml
from .datasets import load_workload
from .pricing import PriceBook
from .results import ResultWriter
from .scoring import score_sync


async def run_quality(
    pairs: list[tuple[str, System, str]],      # (label, system, model id on that system)
    cfg_path: str | Path,
    out_path: str | Path,
    prices_path: str | Path = "configs/prices.yaml",
    temperature: float | None = 0.0,
    log=print,
    progress=None,
    seed: int = 0,
) -> Path:
    cfg = load_yaml(cfg_path)
    prices = PriceBook.load(prices_path)
    items_by_set: dict[str, list[dict[str, Any]]] = {}
    for st in cfg.get("sets", []):
        p = Path(st["path"])
        if not p.exists():
            log(f"[skip] {st['name']}: {p} not found (download it from the Quality parity panel or run `gbench prepare-data`)")
            continue
        items_by_set[st["name"]] = load_workload(p, st.get("max_items"))
    if not items_by_set:
        raise SystemExit("No question sets found for the quality test. Download them first (needs `pip install -e '.[data]'`).")

    reps = int(cfg.get("repeats", 1))
    jobs = [(name, it, rep, lab, sys_, model) for name, items in items_by_set.items() for it in items
            for rep in range(reps) for (lab, sys_, model) in pairs]
    random.Random(seed).shuffle(jobs)
    writer = ResultWriter(Path(out_path), {"track": "Q"})
    sem = asyncio.Semaphore(int(cfg.get("concurrency", 8)))
    max_tokens = int(cfg.get("max_tokens", 4096))
    done = 0
    guard = CreditGuard()

    async with make_client() as client:
        async def one(job):
            nonlocal done
            name, it, rep, lab, sys_, model = job
            if guard.tripped:
                return
            msgs = ([{"role": "system", "content": it["system"]}] if it.get("system") else []) + \
                   [{"role": "user", "content": it["prompt"]}]
            async with sem:
                r = await chat(client, sys_, model, msgs, max_tokens=max_tokens, temperature=temperature,
                               stream=True, seed=None,
                               meta={"set": name, "item_id": it["id"], "label": lab, "rep": rep})
            correct = None
            if r.ok:
                try:
                    correct = score_sync(it.get("scorer", "mc"), r.text, it)
                except RuntimeError as e:
                    log(f"[warn] {e}")
            guard.record(r)
            rec = r.to_dict()
            cost, src = prices.cost(rec)
            rec.update({"correct": correct, "cost_usd": cost, "cost_source": src, "answer": it.get("answer"),
                        "text": r.text[-2000:]})
            writer.write(rec)
            done += 1
            if progress:
                progress(done, len(jobs))
            if done % 50 == 0 or done == len(jobs):
                log(f"  quality: {done}/{len(jobs)}")

        await asyncio.gather(*(one(j) for j in jobs))

    writer.close()
    guard.check()
    return Path(out_path)
