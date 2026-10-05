"""Track B: routing quality per dollar across workloads."""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from pathlib import Path

from .client import CreditGuard, chat, make_client
from .config import System, load_yaml
from .datasets import load_workload
from .pricing import PriceBook
from .results import ResultWriter
from .scoring import JUDGE_PROMPT, parse_judge, score_sync


@dataclass
class Config:
    label: str        # name used in results, e.g. "concentrate" or "direct-openai:frontier"
    system: System
    model: str
    role: str         # "router" | "baseline"


def build_configs(systems: dict[str, System], cfg: dict, only: list[str] | None, log=print) -> list[Config]:
    out: list[Config] = []
    for name in cfg.get("routers", []):
        s = systems.get(name)
        if not s or (only and name not in only):
            continue
        if not s.available:
            log(f"[skip] {name}: env var {s.api_key_env} is not set")
            continue
        if not s.router_model:
            log(f"[skip] {name}: no router_model configured")
            continue
        out.append(Config(name, s, s.router_model, "router"))
    for spec in cfg.get("baselines", []):
        name, alias = spec.split(":")
        s = systems.get(name)
        if not s or (only and spec not in only and name not in only):
            continue
        if not s.available:
            log(f"[skip] {spec}: env var {s.api_key_env} is not set")
            continue
        if alias not in s.models:
            log(f"[skip] {spec}: alias not mapped")
            continue
        out.append(Config(spec, s, s.models[alias], "baseline"))
    return out


async def run_track_b(
    systems: dict[str, System],
    cfg_path: str | Path = "configs/track_b.yaml",
    out_dir: str | Path = "results",
    only: list[str] | None = None,
    workloads_filter: list[str] | None = None,
    max_items: int | None = None,
    prices_path: str | Path = "configs/prices.yaml",
    seed: int = 0,
    log=print,
    progress=None,
    out_path: str | Path | None = None,
) -> Path:
    cfg = load_yaml(cfg_path)
    configs = build_configs(systems, cfg, only, log)
    if not configs:
        raise SystemExit("No Track B configs available (check API keys and track_b.yaml).")
    prices = PriceBook.load(prices_path)

    judge_sys = None
    jcfg = cfg.get("judge") or {}
    if jcfg:
        js = systems.get(jcfg.get("system", ""))
        if js and js.available and jcfg.get("alias") in js.models:
            judge_sys = (js, js.models[jcfg["alias"]])
        else:
            log("[warn] judge system unavailable; scorer=judge items will be left unscored")

    items_by_wl = {}
    for wl in cfg.get("workloads", []):
        if workloads_filter and wl["name"] not in workloads_filter:
            continue
        p = Path(wl["path"])
        if not p.exists():
            log(f"[skip] workload {wl['name']}: {p} not found (run `gbench prepare-data`)")
            continue
        items_by_wl[wl["name"]] = load_workload(p, max_items or wl.get("max_items"))

    jobs = [(wl, it, c) for wl, items in items_by_wl.items() for it in items for c in configs]
    random.Random(seed).shuffle(jobs)

    out = Path(out_path) if out_path else Path(out_dir) / f"track_b_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    writer = ResultWriter(out, {"track": "B"})
    sem = asyncio.Semaphore(int(cfg.get("concurrency", 16)))
    max_tokens = int(cfg.get("max_tokens", 1024))
    temperature = float(cfg.get("temperature", 0))
    done = 0

    guard = CreditGuard()
    async with make_client() as client:
        async def one(job):
            nonlocal done
            wl, it, c = job
            if guard.tripped:
                return
            msgs = ([{"role": "system", "content": it["system"]}] if it.get("system") else []) + \
                   [{"role": "user", "content": it["prompt"]}]
            async with sem:
                r = await chat(client, c.system, c.model, msgs, max_tokens=max_tokens,
                               temperature=temperature, stream=True,
                               meta={"workload": wl, "item_id": it["id"], "label": c.label, "role": c.role})
                correct = None
                judge_cost = None
                if r.ok:
                    scorer = it.get("scorer", "judge")
                    if scorer == "judge":
                        if judge_sys:
                            js, jm = judge_sys
                            jr = await chat(client, js, jm, [{"role": "user", "content": JUDGE_PROMPT.format(
                                question=it["prompt"], reference=it.get("answer", ""), candidate=r.text)}],
                                max_tokens=8, temperature=0, stream=False)
                            correct = parse_judge(jr.text) if jr.ok else None
                            judge_cost, _ = prices.cost(jr.to_dict())
                    else:
                        try:
                            correct = score_sync(scorer, r.text, it)
                        except RuntimeError as e:
                            log(f"[warn] {e}")
            guard.record(r)
            rec = r.to_dict()
            cost, src = prices.cost(rec)
            rec.update({"correct": correct, "cost_usd": cost, "cost_source": src, "judge_cost_usd": judge_cost,
                        "text": r.text[:4000]})
            writer.write(rec)
            done += 1
            if progress:
                progress(done, len(jobs))
            if done % 50 == 0 or done == len(jobs):
                log(f"  track B: {done}/{len(jobs)}")

        await asyncio.gather(*(one(j) for j in jobs))

    writer.close()
    guard.check()
    log(f"Track B results -> {out}")
    return out
