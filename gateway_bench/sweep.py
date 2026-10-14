"""Scheduled sweep over the top-10 models: speed, volume, cache and quality, 3 times a day.

    gbench sweep --dry-run          # what would run now, and its estimated cost
    gbench sweep                    # run the current slot (morning / afternoon / evening by local time)
    gbench price-check              # per-provider prices on both gateways for the 10 models
    gbench schedule install         # run `gbench sweep` 3x a day on this Mac (launchd)

Each model x routing mode becomes one run in results/ui/ (so it shows in the web UI), and one row per
gateway x test x cell is appended to results/sweeps/sweep_log.csv, the sheet to open in Sheets or Excel.
"""
from __future__ import annotations

import asyncio
import csv
import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from .config import load_yaml

REF, RIVAL = "concentrate", "openrouter"
SWEEP_DIR = Path("results") / "sweeps"
LOG_CSV = SWEEP_DIR / "sweep_log.csv"
DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

# Concentrate provider name -> OpenRouter provider slug (for pinning OpenRouter to the same upstream)
PROVIDER_SLUGS = {
    "vertex": "google-vertex", "ai-studio": "google-ai-studio", "bedrock": "amazon-bedrock", "azure": "azure",
    "zai": "z-ai", "xai": "x-ai", "moonshot": "moonshotai", "alibaba": "alibaba", "openai": "openai",
    "anthropic": "anthropic", "deepseek": "deepseek", "deepinfra": "deepinfra", "novita": "novita",
    "fireworks": "fireworks", "baseten": "baseten", "together": "together", "cloudflare": "cloudflare",
    "crusoe": "crusoe", "wafer": "wafer", "modal": "modal", "nebius": "nebius", "runware": "runware",
    "azure-fw": "azure",
}

CSV_COLS = ["sweep_id", "date", "local_time", "weekday", "slot", "model", "mode", "pinned_to", "gateway", "test", "cell",
            "n", "ok", "success_pct", "ttft_p50_ms", "ttft_p90_ms", "ttft_p99_ms", "first_answer_p50_ms",
            "e2e_p50_ms", "e2e_p99_ms", "tps_p50", "rps", "in_tok_mean", "out_tok_mean", "reasoning_tok_mean",
            "cached_share_pct", "cost_per_req_usd", "cost_source", "providers_served", "errors", "run_id"]


def _norm(x: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def current_slot(now: dt.datetime | None = None) -> str:
    h = (now or dt.datetime.now()).hour
    return "morning" if h < 11 else ("afternoon" if h < 17 else "evening")


def is_when(when: str | None, slot: str, cfg: dict, now: dt.datetime) -> bool:
    if not when or when == "always":
        return True
    if when == "daily":
        return slot == (cfg.get("daily") or {}).get("slot", "afternoon")
    if when == "weekly":
        wk = cfg.get("weekly") or {}
        return slot == wk.get("slot", "evening") and DAYS[now.weekday()] == wk.get("day", "sun")
    return False


# ------------------------------------------------------------------ prices

def tokens_for(shape: dict) -> tuple[float, float]:
    """Measured input is ~25% above the target (system prompt + filler tokenisation)."""
    return shape["input_tokens"] * 1.25 + 30, shape["output_tokens"]


def blended(p: dict | None) -> float | None:
    return None if not p else (4 * p["input"] + p["output"]) / 5


def resolve_models(cfg: dict, ref_models: list[dict], or_catalog: list[dict], log=print) -> list[dict]:
    """Attach Concentrate's entry and the confirmed OpenRouter id to each configured model."""
    from .ui.server import match_ref_model
    or_ids = {m["id"]: m for m in or_catalog}
    out = []
    for m in cfg["models"]:
        ref = match_ref_model(m["concentrate"], ref_models) if ref_models else None
        or_id = m["openrouter"]
        if or_catalog and or_id not in or_ids:
            cand = [x for x in or_catalog if ":" not in x["id"] and
                    (_norm(x.get("name", "").split(":", 1)[-1]) == _norm(m["name"]) or _norm(x["id"].split("/", 1)[-1]) == _norm(m["concentrate"]))]
            if cand:
                log(f"[info] {m['name']}: OpenRouter id {or_id} not found; using {cand[0]['id']}")
                or_id = cand[0]["id"]
            else:
                log(f"[warn] {m['name']}: {or_id} not in OpenRouter's catalog; runs may fail their pre-run check")
        if ref_models and not ref:
            log(f"[warn] {m['name']}: {m['concentrate']} not in Concentrate's model list")
        out.append({**m, "openrouter": or_id, "ref": ref, "or_entry": or_ids.get(or_id)})
    return out


def or_endpoints(model_id: str) -> list[dict]:
    from .ui.claims import parse_endpoints
    from .ui.server import _or_get
    author, slug = model_id.split("/", 1)
    eps = ((_or_get(f"/models/{author}/{slug}/endpoints").get("data") or {}).get("endpoints")) or []
    rows = parse_endpoints(eps)
    for ep, row in zip(eps, rows):
        pr = ep.get("pricing") or {}
        try:
            row["input"], row["output"] = float(pr.get("prompt")) * 1e6, float(pr.get("completion")) * 1e6
        except (TypeError, ValueError):
            row["input"] = row["output"] = None
        row["cache_read"] = float(pr["input_cache_read"]) * 1e6 if pr.get("input_cache_read") else None
        row["discount"] = ep.get("discount") or pr.get("discount")
    return rows


def price_check(cfg_path: str = "configs/top10.yaml", out_csv: str | None = None, log=print) -> list[dict]:
    """Per model and provider: Concentrate's list price vs OpenRouter's price for the same provider."""
    from .ui.server import fetch_catalog, fetch_ref_models
    cfg = load_yaml(cfg_path)
    refs, err = fetch_ref_models(force=True)
    if err:
        log(f"[warn] {err}")
    cat, err2 = fetch_catalog(force=True)
    if err2:
        log(f"[warn] {err2}")
    rows = []
    for m in resolve_models(cfg, refs, cat, log):
        try:
            eps = or_endpoints(m["openrouter"])
        except Exception as e:  # noqa: BLE001
            log(f"[warn] {m['name']}: OpenRouter endpoints unavailable ({e})")
            eps = []
        by_slug = {}
        for e in eps:
            by_slug.setdefault(e["tag"].split("/")[0], e)
        cheapest_or = min((e for e in eps if e.get("input") is not None), key=lambda e: blended(e), default=None)
        for prov, cp in sorted(((m["ref"] or {}).get("provider_prices") or {}).items(), key=lambda kv: blended(kv[1])):
            e = by_slug.get(PROVIDER_SLUGS.get(prov, prov))
            rows.append({"model": m["name"], "provider": prov,
                         "conc_in": cp["input"], "conc_out": cp["output"],
                         "or_in": e["input"] if e else None, "or_out": e["output"] if e else None,
                         "or_cache_read": e["cache_read"] if e else None, "or_quant": e.get("quantization") if e else None,
                         "conc_vs_or_same_provider": (blended(cp) / blended(e)) if (e and blended(e)) else None,
                         "conc_vs_or_default": (blended(cp) / blended(cheapest_or)) if cheapest_or and blended(cheapest_or) else None,
                         "or_default_provider": cheapest_or["provider"] if cheapest_or else None})
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["model"])
            w.writeheader()
            w.writerows(rows)
        log(f"price check -> {out_csv}")
    return rows


# ------------------------------------------------------------------ plan

def plan(cfg: dict, slot: str, now: dt.datetime, models: list[dict]) -> list[dict]:
    """Units to run this slot (model x mode) with their tests and estimated cost."""
    shapes = {s["name"]: s for s in load_yaml("configs/track_a.yaml")["shapes"]}
    sp = cfg.get("speed") or {}
    vol = cfg.get("volume") or {}
    cp = cfg.get("cache_probe") or {}
    q = cfg.get("quality") or {}
    units = []
    for m in models:
        price = (m.get("ref") or {}).get("price") or {"input": 0.0, "output": 0.0}
        dear = max(((m.get("ref") or {}).get("provider_prices") or {"x": price}).values(), key=blended)

        def cost(n, shape, p=dear):
            i, o = tokens_for(shape)
            return n * (i * p["input"] + o * p["output"]) / 1e6

        for mode in cfg.get("modes", ["same_provider", "default"]):
            u: dict[str, Any] = {"model": m, "mode": mode, "tests": [], "est_usd": 0.0}
            n_cell = int(sp.get("requests_per_cell", 10)) * len(sp.get("stream_modes", [True]))
            sh_names = sp.get("shapes", list(shapes))
            u["speed"] = {"shapes": sh_names, "n": int(sp.get("requests_per_cell", 10)), "warmup": int(sp.get("warmup", 5)),
                          "stream_modes": sp.get("stream_modes", [True])}
            u["tests"].append("speed")
            u["est_usd"] += 2 * sum(cost(n_cell, shapes[s]) for s in sh_names) + 2 * cost(int(sp.get("warmup", 5)), shapes[sh_names[0]])
            levels = []
            for lv in vol.get("levels", []):
                if not is_when(lv.get("when"), slot, cfg, now):
                    continue
                if lv.get("max_price_per_m") is not None and (blended(price) or 0) > float(lv["max_price_per_m"]):
                    continue
                levels.append({"requests": int(lv["requests"]), "concurrency": int(lv["concurrency"])})
            if levels:
                u["volume"] = {"levels": levels, "shape": vol.get("shape", "tiny")}
                u["tests"].append("volume " + "/".join(f"{lv['requests']:,}" for lv in levels))
                u["est_usd"] += 2 * sum(cost(lv["requests"], shapes[vol.get("shape", "tiny")]) for lv in levels)
            if cp and is_when(cp.get("when", "daily"), slot, cfg, now):
                u["cache"] = {"prefix_tokens": int(cp.get("prefix_tokens", 8000))}
                u["tests"].append("cache")
                u["est_usd"] += 2 * 4 * (u["cache"]["prefix_tokens"] * 1.05 * dear["input"]) / 1e6
            if q and is_when(q.get("when", "weekly"), slot, cfg, now) and mode in q.get("modes", ["same_provider"]):
                u["quality"] = {"sets": q.get("sets", ["gpqa_diamond"]), "items": int(q.get("items", 198))}
                u["tests"].append("quality")
                u["est_usd"] += 2 * u["quality"]["items"] * (600 * dear["input"] + 4000 * dear["output"]) / 1e6
            units.append(u)
    return units


def apply_budget(units: list[dict], budget: float, log=print) -> list[dict]:
    """Drop the most expensive optional tests first (quality, then the largest volume level), then whole units."""
    total = sum(u["est_usd"] for u in units)
    if total <= budget:
        return units
    log(f"[budget] estimated ${total:.2f} is over ${budget:.2f}; trimming the most expensive units")
    keep = sorted(units, key=lambda u: u["est_usd"])
    out, spent = [], 0.0
    for u in keep:
        if spent + u["est_usd"] <= budget:
            out.append(u)
            spent += u["est_usd"]
        else:
            log(f"[budget] skipped {u['model']['name']} ({u['mode']}), est ${u['est_usd']:.2f}")
    return [u for u in units if u in out]


# ------------------------------------------------------------------ run

async def _served_provider_now(model_id: str) -> str | None:
    """Ask Concentrate once which provider it serves this model from right now."""
    from .client import chat, make_client
    from .config import load_systems
    s = load_systems("configs/systems.yaml")[REF]
    async with make_client(4) as client:
        r = await chat(client, s, model_id, [{"role": "user", "content": "Reply with the word OK."}],
                       max_tokens=8, stream=False, seed=None, temperature=None)
    return r.provider_served if r.ok else None


async def run_unit(u: dict, slot: str, sweep_id: str, log=print) -> tuple[str | None, str | None]:
    from .ui import server
    m = u["model"]
    pin = None
    if u["mode"] == "same_provider":
        prov = await _served_provider_now(m["concentrate"])
        if prov:
            slug = PROVIDER_SLUGS.get(prov, prov)
            try:
                tags = {e["tag"].split("/")[0] for e in or_endpoints(m["openrouter"])}
            except Exception as e:  # noqa: BLE001
                tags = set()
                log(f"[warn] {m['name']}: couldn't read OpenRouter's providers ({e})")
            if slug in tags or not tags:
                pin = slug
            else:
                log(f"[warn] {m['name']}: Concentrate served it from {prov}, which OpenRouter doesn't offer; "
                    "running unpinned and recording it")
        else:
            log(f"[warn] {m['name']}: Concentrate didn't say which provider it used; running unpinned")
    opts = server.RunOptions(
        mode="live", label=f"sweep {slot} · {m['name']} · {u['mode'].replace('_', ' ')}",
        track_a=True, track_b=False, load=False, include_baseline=False,
        shapes=u["speed"]["shapes"], stream_modes=u["speed"]["stream_modes"],
        requests_per_cell=u["speed"]["n"], warmup=u["speed"]["warmup"],
        model=m["openrouter"], ref_model=m["concentrate"], upstream=pin or "",
        quality="quality" in u, quality_sets=(u.get("quality") or {}).get("sets", []),
        quality_items=(u.get("quality") or {}).get("items", 50),
        volume="volume" in u, volume_levels=(u.get("volume") or {}).get("levels", [{"requests": 100, "concurrency": 10}]),
        volume_shape=(u.get("volume") or {}).get("shape", "tiny"),
        cache_probe="cache" in u, cache_prefix_tokens=(u.get("cache") or {}).get("prefix_tokens", 8000),
        sweep={"sweep_id": sweep_id, "slot": slot, "pin_mode": u["mode"], "model_name": m["name"], "pinned_to": pin})
    run = server.Run(opts)
    server.RUNS[run.id] = run
    run.save()
    log(f"== {m['name']} · {u['mode']}" + (f" (OpenRouter pinned to {pin})" if pin else "") + f" · run {run.id}")
    await server.execute(run)
    log(f"   {run.status}" + (f": {run.error}" if run.error else ""))
    return run.id, pin


def log_rows(run_id: str, u: dict, slot: str, sweep_id: str, pin: str | None, now: dt.datetime) -> list[dict]:
    """One row per gateway x test x cell, from the run's raw results."""
    import numpy as np
    from .results import read_results
    from .pricing import PriceBook
    from .ui import server
    from .ui.summary import _error_kinds, fill_costs
    run_dir = server.RUNS_DIR / run_id
    files = sorted(run_dir.glob("*.jsonl"))
    if not files:
        return []
    df = read_results(files)
    if df.empty:
        return []
    for c in ("meta.warmup", "meta.shape", "meta.level", "meta.set", "meta.repeat", "provider_served", "cached_tokens",
              "reasoning_tokens", "first_text_ms", "output_tps", "cost_reported", "model_served"):
        if c not in df:
            df[c] = np.nan
    prices = run_dir / "prices.yaml" if (run_dir / "prices.yaml").exists() else Path("configs/prices.yaml")
    df = fill_costs(df, PriceBook.load(prices), server.build_catalog("live"))
    df = df[df["meta.warmup"] != True]  # noqa: E712
    df = df[df.system.isin([REF, RIVAL])]
    rows = []
    base = {"sweep_id": sweep_id, "date": now.strftime("%Y-%m-%d"), "local_time": now.strftime("%H:%M"),
            "weekday": DAYS[now.weekday()], "slot": slot, "model": u["model"]["name"], "mode": u["mode"],
            "pinned_to": pin or "", "run_id": run_id}

    def q(s, p):
        s = s.dropna()
        return round(float(s.quantile(p)), 1) if len(s) else None

    def add(test, cell, g):
        ok = g[g.ok == True]  # noqa: E712
        wall = None
        if test == "volume" and len(g):
            wall = float((g.t_start + g.e2e_ms.fillna(0) / 1000).max() - g.t_start.min())
        it = ok.input_tokens.fillna(0).sum()
        rows.append({**base, "gateway": g.system.iloc[0], "test": test, "cell": cell, "n": int(len(g)), "ok": int(len(ok)),
                     "success_pct": round(len(ok) / len(g) * 100, 2) if len(g) else None,
                     "ttft_p50_ms": q(ok.ttft_ms, .5), "ttft_p90_ms": q(ok.ttft_ms, .9), "ttft_p99_ms": q(ok.ttft_ms, .99),
                     "first_answer_p50_ms": q(ok.first_text_ms, .5), "e2e_p50_ms": q(ok.e2e_ms, .5), "e2e_p99_ms": q(ok.e2e_ms, .99),
                     "tps_p50": q(ok.output_tps, .5), "rps": round(len(ok) / wall, 2) if wall else None,
                     "in_tok_mean": round(float(ok.input_tokens.mean()), 1) if ok.input_tokens.notna().any() else None,
                     "out_tok_mean": round(float(ok.output_tokens.mean()), 1) if ok.output_tokens.notna().any() else None,
                     "reasoning_tok_mean": round(float(ok.reasoning_tokens.mean()), 1) if ok.reasoning_tokens.notna().any() else None,
                     "cached_share_pct": round(float(ok.cached_tokens.fillna(0).sum() / it * 100), 1) if (it and ok.cached_tokens.notna().any()) else None,
                     "cost_per_req_usd": round(float(ok.cost_usd.mean()), 8) if ok.cost_usd.notna().any() else None,
                     "cost_source": ",".join(sorted(set(ok.cost_source.dropna().astype(str)))),
                     "providers_served": ";".join(f"{k}:{v}" for k, v in ok.provider_served.fillna("?").value_counts().items()),
                     "errors": ";".join(f"{k}:{v}" for k, v in _error_kinds(g[g.ok != True]).items())})  # noqa: E712

    for (track, sys_), g in df.groupby(["track", "system"]):
        if track == "A":
            for shape, gs in g.groupby("meta.shape"):
                add("speed", str(shape), gs)
        elif track == "V":
            for lv, gs in g.groupby("meta.level"):
                add("volume", str(lv), gs)
        elif track == "C":
            add("cache", "repeats", g[g["meta.repeat"].fillna(0) > 0])
        elif track == "Q":
            for st, gs in g.groupby("meta.set"):
                ok = gs[gs.correct.notna()]
                add("quality", str(st), gs)
                rows[-1]["success_pct"] = round(float(ok.correct.astype(bool).mean() * 100), 2) if len(ok) else None
                rows[-1]["cell"] = f"{st} (success_pct = accuracy)"
    return rows


def append_csv(rows: list[dict], path: Path = LOG_CSV) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


async def run_sweep(cfg_path: str = "configs/top10.yaml", slot: str | None = None, only: list[str] | None = None,
                    modes: list[str] | None = None, dry_run: bool = False, log=print) -> list[dict]:
    from .ui.server import fetch_catalog, fetch_ref_models
    now = dt.datetime.now()
    cfg = load_yaml(cfg_path)
    slot = slot or current_slot(now)
    refs, err = fetch_ref_models(force=True)
    if err:
        log(f"[warn] {err}")
    cat, err2 = fetch_catalog(force=True)
    if err2:
        log(f"[warn] {err2}")
    models = resolve_models(cfg, refs, cat, log)
    if only:
        models = [m for m in models if any(_norm(o) in _norm(m["name"]) for o in only)]
    if modes:
        cfg["modes"] = modes
    units = plan(cfg, slot, now, models)
    weekly = is_when("weekly", slot, cfg, now)
    budget = float(cfg.get("budget_usd_weekly_slot" if weekly else "budget_usd_per_slot", 60))
    log(f"Sweep · {slot}{' (weekly slot)' if weekly else ''} · {now:%Y-%m-%d %H:%M} · {len(units)} units · "
        f"estimated ${sum(u['est_usd'] for u in units):.2f} at Concentrate's dearest provider price (budget ${budget:.0f})")
    for u in units:
        log(f"  {u['model']['name']:<24} {u['mode']:<14} {', '.join(u['tests']):<40} ~${u['est_usd']:.2f}")
    units = apply_budget(units, budget, log)
    if dry_run:
        return units
    sweep_id = now.strftime("%Y%m%d-") + slot
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    for u in units:
        try:
            rid, pin = await run_unit(u, slot, sweep_id, log)
            append_csv(log_rows(rid, u, slot, sweep_id, pin, now))
        except Exception as e:  # noqa: BLE001 - one model failing must not stop the sweep
            log(f"[error] {u['model']['name']} ({u['mode']}): {e}")
    log(f"Sweep done -> {LOG_CSV}")
    return units


# ------------------------------------------------------------------ schedule (macOS launchd)

LABEL = "ai.concentrate.gbench-sweep"


def plist_text(slots: dict, repo: Path, gbench: str, log_path: Path) -> str:
    from xml.sax.saxutils import escape as esc
    cal = ""
    for t in slots.values():
        hh, mm = (int(x) for x in str(t).split(":"))
        cal += f"    <dict><key>Hour</key><integer>{hh}</integer><key>Minute</key><integer>{mm}</integer></dict>\n"
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{LABEL}</string>
  <key>ProgramArguments</key>
  <array><string>{esc(gbench)}</string><string>sweep</string></array>
  <key>WorkingDirectory</key><string>{esc(str(repo))}</string>
  <key>StartCalendarInterval</key>
  <array>
{cal}  </array>
  <key>StandardOutPath</key><string>{esc(str(log_path))}</string>
  <key>StandardErrorPath</key><string>{esc(str(log_path))}</string>
</dict>
</plist>
"""


def schedule(action: str, cfg_path: str = "configs/top10.yaml", log=print) -> int:
    import shutil
    import subprocess
    cfg = load_yaml(cfg_path)
    slots = cfg.get("slots") or {"morning": "08:47", "afternoon": "13:47", "evening": "19:47"}
    plist = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
    repo = Path.cwd().resolve()
    if action == "show":
        gb = shutil.which("gbench") or sys.argv[0]
        print(plist_text(slots, repo, gb, repo / SWEEP_DIR / "schedule.log"))
        return 0
    if sys.platform != "darwin":
        log("Scheduling uses macOS launchd. On Linux add this to `crontab -e` instead:")
        for t in slots.values():
            hh, mm = str(t).split(":")
            log(f"  {int(mm)} {int(hh)} * * *  cd {repo} && {shutil.which('gbench') or 'gbench'} sweep >> {repo / SWEEP_DIR / 'schedule.log'} 2>&1")
        return 1
    if action == "install":
        gb = shutil.which("gbench")
        if not gb:
            log("Can't find the gbench command; activate the venv first.")
            return 1
        (repo / SWEEP_DIR).mkdir(parents=True, exist_ok=True)
        plist.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["launchctl", "unload", str(plist)], capture_output=True)
        plist.write_text(plist_text(slots, repo, gb, repo / SWEEP_DIR / "schedule.log"))
        r = subprocess.run(["launchctl", "load", str(plist)], capture_output=True, text=True)
        if r.returncode:
            log(f"launchctl load failed: {r.stderr.strip()}")
            return 1
        log(f"Scheduled: gbench sweep at {', '.join(f'{k} {v}' for k, v in slots.items())} (local time).")
        log("If the Mac is asleep at a slot, launchd runs it once when it wakes. Log: results/sweeps/schedule.log")
        return 0
    if action == "uninstall":
        subprocess.run(["launchctl", "unload", str(plist)], capture_output=True)
        if plist.exists():
            plist.unlink()
        log("Schedule removed.")
        return 0
    if action == "status":
        r = subprocess.run(["launchctl", "list"], capture_output=True, text=True)
        on = LABEL in r.stdout
        log(f"Schedule {'installed' if on else 'not installed'}" + (f": {plist}" if on else ""))
        if LOG_CSV.exists():
            log(f"Last sweep rows: {sum(1 for _ in open(LOG_CSV)) - 1} in {LOG_CSV}")
        return 0
    log("actions: install | uninstall | status | show")
    return 1
