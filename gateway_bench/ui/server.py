"""Local web UI: configure and launch Concentrate-vs-OpenRouter runs, watch progress, view results.

Start with:  gbench ui            (http://127.0.0.1:8765)
API keys are read from .env on this machine and never sent to the browser.
"""
from __future__ import annotations

import asyncio
import copy
import re
import json
import os
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from ..config import System, fill_pin, load_systems, load_yaml
from .summary import build_summary, clean, readable_error

REF, RIVAL, BASELINE = "concentrate", "openrouter", "direct-openai"
STATIC = Path(__file__).parent / "static"
ROOT = Path.cwd()
RUNS_DIR = ROOT / "results" / "ui"


def _cfg_dir(mode: str) -> Path:
    return ROOT / ("configs/ui_demo" if mode == "demo" else "configs")


# ------------------------------------------------------------------ demo mocks

_mock_lock = threading.Lock()
_mocks_started = False
DEMO_MOCKS = [  # port, latency ms, accuracy, $/1M tok, name   (synthetic, for trying the UI only)
    (9301, 40, 0.62, 0.6, "direct-openai"),
    (9302, 70, 0.95, 12.0, "direct-frontier"),
    (9303, 46, 0.90, 3.1, "concentrate"),
    (9304, 68, 0.86, 4.4, "openrouter"),
]


def ensure_demo_mocks() -> None:
    global _mocks_started
    with _mock_lock:
        if _mocks_started:
            return
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        from mock_gateway import serve  # type: ignore
        for port, lat, acc, price, name in DEMO_MOCKS:
            try:
                srv = serve(port, latency_ms=lat, tok_ms=1, fail_rate=0.01, accuracy=acc, price_per_mtok=price, name=name)
            except OSError:
                # port taken: most likely another `gbench ui` already serves the demo mocks; use those
                continue
            threading.Thread(target=srv.serve_forever, daemon=True).start()
        os.environ.setdefault("GBENCH_DEMO_KEY", "demo")
        _mocks_started = True


# ------------------------------------------------------------------ run state


class RunOptions(BaseModel):
    mode: str = Field("demo", pattern="^(demo|live)$")
    label: str = ""
    track_a: bool = True
    track_b: bool = True
    load: bool = False
    include_baseline: bool = True
    shapes: list[str] = []
    stream_modes: list[bool] = [True, False]
    requests_per_cell: int = Field(20, ge=1, le=5000)
    warmup: int = Field(3, ge=0, le=200)
    workloads: list[str] = []
    max_items: int = Field(50, ge=1, le=5000)
    include_model_baselines: bool = True
    rates: list[float] = [5, 20]
    step_duration_s: float = Field(10, ge=1, le=3600)
    # Model for Speed & fees and Load. Empty = the config default (model_alias in track_a.yaml).
    model: str = ""            # catalog id, "author/model" (OpenRouter style)
    upstream: str = ""         # provider slug to pin, e.g. "novita"; empty = gateway decides
    direct_model: str = ""     # id for the direct-provider baseline; empty = derived
    ref_model: str = ""        # Concentrate's own id for the model, when it names it differently
    judge_model: str = ""      # grader for open-ended questions, called through OpenRouter; empty = skip them


class Run:
    def __init__(self, opts: RunOptions):
        self.id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        self.opts = opts
        self.dir = RUNS_DIR / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.status = "queued"
        self.created = time.time()
        self.started: float | None = None
        self.finished: float | None = None
        self.error: str | None = None
        self.phases: dict[str, dict[str, Any]] = {}
        self.logs: list[str] = []
        self.task: asyncio.Task | None = None
        self._last_save = 0.0
        self.models: dict[str, Any] = {}
        self.preflight: list[dict[str, Any]] = []

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')}  {msg.strip()}"
        self.logs.append(line)
        self.logs = self.logs[-400:]

    def progress(self, phase: str):
        def cb(done: int, total: int):
            self.phases[phase].update(done=done, total=total)
            if time.time() - self._last_save > 2:
                self.save()
        return cb

    def state(self) -> dict:
        return clean({"id": self.id, "status": self.status, "created": self.created, "started": self.started,
                      "finished": self.finished, "error": self.error, "phases": self.phases,
                      "options": self.opts.model_dump(), "models": self.models,
                      "preflight": self.preflight, "logs": self.logs[-60:]})

    def save(self) -> None:
        self._last_save = time.time()
        (self.dir / "state.json").write_text(json.dumps(self.state(), indent=2))


RUNS: dict[str, Run] = {}


def _load_saved_runs() -> list[dict]:
    out = []
    if RUNS_DIR.exists():
        for p in RUNS_DIR.glob("*/state.json"):
            try:
                st = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            if st.get("status") in ("running", "queued") and st["id"] not in RUNS:
                st["status"] = "interrupted"
            out.append(st)
    return out


# ------------------------------------------------------------------ run execution


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def load_providers() -> dict:
    p = ROOT / "configs" / "providers.yaml"
    return load_yaml(p) if p.exists() else {"providers": {}, "picker_authors": []}


def derive_direct_id(model: str, rule: str) -> str:
    return model.split("/", 1)[1] if rule == "strip_author" and "/" in model else model


def resolve_model_setup(o: "RunOptions", systems: dict[str, System], mode: str, base_alias: str,
                        base_baseline: str | None, log) -> tuple[str, list[System], System | None, dict]:
    """Return (alias, [ref, rival] copies with model + pin applied, baseline system or None, info)."""
    gw = [copy.deepcopy(systems[REF]), copy.deepcopy(systems[RIVAL])]
    info: dict[str, Any] = {"model": o.model or None, "upstream": o.upstream or None, "ids": {}, "warnings": []}
    if not o.model:
        alias = base_alias
        base = systems.get(base_baseline) if base_baseline else None
        for s_ in gw + ([base] if base else []):
            if alias in s_.models:
                info["ids"][s_.name] = s_.models[alias]
        return alias, gw, (base if (base and base.available) else None), info

    alias = "__run__"
    ref_entry = None
    if mode == "live":
        refs, _ = fetch_ref_models()
        ref_entry = match_ref_model(o.ref_model.strip() or o.model, refs)
    ref_id = o.ref_model.strip() or (ref_entry["id"] if ref_entry else o.model)
    if not o.ref_model.strip() and ref_entry and ref_id != o.model:
        log(f"[info] Concentrate's name for {o.model} is {ref_id} (from its model list)")
    info["ref_entry"] = ref_entry
    for s_ in gw:
        s_.models[alias] = ref_id if s_.name == REF else o.model
        info["ids"][s_.name] = s_.models[alias]
        if o.upstream:
            if s_.pin:
                s_.extra_body = {**s_.extra_body, **fill_pin(s_.pin, o.upstream)}
            else:
                w = f"{s_.name} has no provider pin configured, so it may serve {o.model} from a different provider"
                info["warnings"].append(w)
                log(f"[warn] {w}")

    provs = load_providers().get("providers", {})
    author = o.model.split("/", 1)[0] if "/" in o.model else ""
    slug = o.upstream or next((k for k, v in provs.items() if author in (v.get("authors") or [])), "")
    baseline = None
    if mode == "demo":
        b = systems.get(base_baseline or "")
        if b:
            baseline = copy.deepcopy(b)
            baseline.models[alias] = o.direct_model or o.model
    elif slug in provs:
        pv = provs[slug]
        direct_id = o.direct_model or derive_direct_id(o.model, pv.get("direct_model", "strip_author"))
        baseline = System(name=f"direct-{slug}", kind="baseline", base_url=pv["base_url"].rstrip("/"),
                          api_key_env=pv["api_key_env"], tracks=["A"], models={alias: direct_id})
        if not baseline.available:
            log(f"[skip] direct {pv['label']} baseline: {pv['api_key_env']} not set")
            baseline = None
    if baseline:
        info["ids"][baseline.name] = baseline.models[alias]
    info["baseline"] = baseline.name if baseline else None
    return alias, gw, baseline, info


async def preflight(run: Run, test: str, checks: list[tuple[System, str, bool]], temperature=0.0) -> list[str]:
    """Send one tiny request per (system, model) with the run's exact settings. Returns names that
    failed. Results go into run.preflight so the page can show them."""
    from ..client import chat, make_client
    failed = []
    async with make_client(max_connections=8) as client:
        results = await asyncio.gather(*(chat(client, s_, model, [{"role": "user", "content": "Reply with the word OK."}],
                                              max_tokens=8, stream=False, seed=None, temperature=temperature)
                                         for s_, model, _ in checks))
    for (s_, model, required), r in zip(checks, results):
        entry = {"test": test, "system": s_.name, "model": model, "ok": r.ok, "status": r.status,
                 "ms": round(r.e2e_ms) if r.e2e_ms else None, "served": r.model_served,
                 "error": readable_error(r.error) if r.error else None, "required": required}
        run.preflight.append(entry)
        if r.ok:
            run.log(f"check ok: {s_.name} {model} ({entry['ms']} ms, served {r.model_served})")
        else:
            run.log(f"check FAILED: {s_.name} {model}: {entry['error']}")
            failed.append(s_.name)
    run.save()
    return failed


def _core_failures(run: Run, threshold: float = 0.05) -> bool:
    """True if Concentrate or OpenRouter failed more than `threshold` of requests in any test."""
    for fp in run.dir.glob("*.jsonl"):
        tot: dict[str, int] = {}
        bad: dict[str, int] = {}
        with open(fp, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if (r.get("meta") or {}).get("warmup"):
                    continue
                who = (r.get("meta") or {}).get("label") or r.get("system")
                if who not in (REF, RIVAL):
                    continue
                tot[who] = tot.get(who, 0) + 1
                bad[who] = bad.get(who, 0) + (0 if r.get("ok") else 1)
        if any(bad[k] / tot[k] > threshold for k in tot if tot[k]):
            return True
    return False


def _fail_msg(run: Run, names: list[str]) -> str:
    bits = []
    for e in run.preflight:
        if e["system"] in names and not e["ok"]:
            bits.append(f"{e['system']} ({e['model']}): {e['error']}")
    return "Stopped before the test: " + " | ".join(bits)


async def execute(run: Run) -> None:
    from ..load import run_load
    from ..track_a import run_track_a
    from ..track_b import run_track_b

    o = run.opts
    cfg_dir = _cfg_dir(o.mode)
    run.status, run.started = "running", time.time()
    run.save()
    try:
        if o.mode == "demo":
            ensure_demo_mocks()
        systems = load_systems(cfg_dir / "systems.yaml")
        for name in (REF, RIVAL):
            if name not in systems:
                raise RuntimeError(f"'{name}' missing from {cfg_dir / 'systems.yaml'}")
            if not systems[name].available:
                raise RuntimeError(f"{systems[name].api_key_env} is not set in .env, so {name} can't be called")

        ta_base = load_yaml(cfg_dir / "track_a.yaml")
        alias, gw, base_sys, info = resolve_model_setup(o, systems, o.mode, ta_base.get("model_alias", "small"),
                                                        ta_base.get("baseline", BASELINE), run.log)
        temperature = ta_base.get("temperature", 0)
        ref_entry = info.pop("ref_entry", None)
        if ref_entry and ref_entry.get("temperature") is False:
            temperature = None
            info["temperature"] = "omitted"
            run.log(f"[info] Concentrate says {ref_entry['id']} doesn't accept temperature; "
                    "leaving it unset for both gateways so the requests stay identical")
        run.models = info
        if o.model and o.mode == "live":
            _write_run_prices(run.dir, o.model, info)
        run.save()

        if o.track_a:
            names_sys = list(gw)
            if o.include_baseline and base_sys:
                names_sys.append(base_sys)
            elif o.include_baseline:
                run.log("[skip] no direct-provider baseline available, so added delay vs direct won't be shown")
            bad = await preflight(run, "Speed & fees", [(s_, s_.model_for(alias), s_.name in (REF, RIVAL)) for s_ in names_sys],
                                  temperature=temperature)
            if any(n in (REF, RIVAL) for n in bad):
                raise RuntimeError(_fail_msg(run, [n for n in bad if n in (REF, RIVAL)]))
            if base_sys and base_sys.name in bad:
                names_sys = [x for x in names_sys if x.name != base_sys.name]
                base_sys = None
                run.log("[skip] direct baseline failed its check; continuing without it")
            cfg = {**ta_base, "baseline": base_sys.name if (o.include_baseline and base_sys and base_sys in names_sys) else None,
                   "model_alias": alias, "stream_modes": o.stream_modes or [True], "temperature": temperature,
                   "requests_per_cell": o.requests_per_cell, "warmup_requests": o.warmup}
            if o.shapes:
                cfg["shapes"] = [s_ for s_ in ta_base["shapes"] if s_["name"] in o.shapes]
            path = _write_yaml(run.dir / "cfg_track_a.yaml", cfg)
            run.phases["Track A: speed & fees"] = {"done": 0, "total": 0, "status": "running"}
            run.log(f"Track A started on {o.model or alias}" + (f" via {o.upstream}" if o.upstream else ""))
            await run_track_a(names_sys, path, run.dir, log=run.log,
                              progress=run.progress("Track A: speed & fees"), out_path=run.dir / "track_a.jsonl")
            run.phases["Track A: speed & fees"]["status"] = "done"
            run.save()

        if o.track_b:
            base = load_yaml(cfg_dir / "track_b.yaml")
            sys_b = copy.deepcopy(systems)
            wls = [w for w in base.get("workloads", []) if not o.workloads or w["name"] in o.workloads]
            cfg = {**base, "workloads": [{**w, "max_items": o.max_items} for w in wls], "routers": [REF, RIVAL],
                   "baselines": base.get("baselines", []) if o.include_model_baselines else []}
            if o.judge_model:
                sys_b[RIVAL].models["__judge__"] = o.judge_model
                cfg["judge"] = {"system": RIVAL, "alias": "__judge__"}
            elif o.mode == "live":
                cfg["judge"] = None
                run.log("[info] no grader chosen: open-ended questions will be left ungraded")
            checks = [(sys_b[REF], sys_b[REF].router_model, True), (sys_b[RIVAL], sys_b[RIVAL].router_model, True)]
            if o.judge_model:
                checks.append((sys_b[RIVAL], o.judge_model, True))
            bad = await preflight(run, "Routing quality", checks)
            if bad:
                raise RuntimeError(_fail_msg(run, bad))
            path = _write_yaml(run.dir / "cfg_track_b.yaml", cfg)
            run.phases["Track B: routing quality"] = {"done": 0, "total": 0, "status": "running"}
            run.log("Track B started")
            await run_track_b(sys_b, path, run.dir, log=run.log, progress=run.progress("Track B: routing quality"),
                              prices_path=run.dir / "prices.yaml" if (run.dir / "prices.yaml").exists() else (
                                  cfg_dir / "prices.yaml" if (cfg_dir / "prices.yaml").exists() else ROOT / "configs/prices.yaml"),
                              out_path=run.dir / "track_b.jsonl")
            run.phases["Track B: routing quality"]["status"] = "done"
            run.save()

        if o.load:
            base = load_yaml(cfg_dir / "load.yaml")
            cfg = {**base, "rates_rps": o.rates or base["rates_rps"], "step_duration_s": o.step_duration_s,
                   "model_alias": alias, "temperature": temperature}
            if not o.track_a:
                bad = await preflight(run, "Load", [(s_, s_.model_for(alias), True) for s_ in gw], temperature=temperature)
                if bad:
                    raise RuntimeError(_fail_msg(run, bad))
            path = _write_yaml(run.dir / "cfg_load.yaml", cfg)
            run.phases["Load test"] = {"done": 0, "total": 0, "status": "running"}
            run.log("Load test started")
            await run_load(gw, path, run.dir, log=run.log,
                           progress=run.progress("Load test"), out_path=run.dir / "load.jsonl")
            run.phases["Load test"]["status"] = "done"

        run.status = "done_with_errors" if _core_failures(run) else "done"
        run.log("Run complete" + (" with errors (see Errors)" if run.status == "done_with_errors" else ""))
    except asyncio.CancelledError:
        run.status = "cancelled"
        run.log("Run cancelled")
    except SystemExit as e:
        run.status, run.error = "failed", str(e)
        run.log(f"Failed: {e}")
    except Exception as e:  # noqa: BLE001
        run.status, run.error = "failed", f"{e}"
        run.log("Failed: " + "".join(traceback.format_exception_only(type(e), e)).strip())
    finally:
        for ph in run.phases.values():
            if ph.get("status") == "running":
                ph["status"] = run.status
        run.finished = time.time()
        run.save()


# ------------------------------------------------------------------ model catalog (OpenRouter, public)

_CATALOG: dict[str, Any] = {"t": 0.0, "data": None, "error": None}
OPENROUTER_API = "https://openrouter.ai/api/v1"


def fetch_catalog(force: bool = False) -> tuple[list[dict], str | None]:
    if not force and _CATALOG["data"] is not None and time.time() - _CATALOG["t"] < 3600:
        return _CATALOG["data"], None
    import httpx
    try:
        r = httpx.get(f"{OPENROUTER_API}/models", timeout=20)
        r.raise_for_status()
        data = r.json().get("data", [])
    except Exception as e:  # noqa: BLE001
        return (_CATALOG["data"] or []), f"Couldn't load the model catalog: {e}"
    _CATALOG.update(t=time.time(), data=data)
    return data, None


def _per_m(v) -> float | None:
    try:
        return float(v) * 1e6
    except (TypeError, ValueError):
        return None


def _write_run_prices(run_dir: Path, model: str, info: dict) -> None:
    """Fallback list prices for this run's model, taken from the catalog (used only when a
    system doesn't report billed cost)."""
    data, _ = fetch_catalog()
    m = next((x for x in data if x.get("id") == model), None)
    base = load_yaml(ROOT / "configs" / "prices.yaml")
    if m:
        pr = m.get("pricing", {})
        price = {"input": _per_m(pr.get("prompt")) or 0.0, "output": _per_m(pr.get("completion")) or 0.0}
        models = dict(base.get("models") or {})
        for mid in {model, model.split("/", 1)[-1], *info.get("ids", {}).values()}:
            models[mid] = price
        base["models"] = models
    _write_yaml(run_dir / "prices.yaml", base)


# ------------------------------------------------------------------ API

app = FastAPI(title="gateway-bench UI", docs_url="/api/docs")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def get_config():
    out: dict[str, Any] = {"ref": REF, "rival": RIVAL, "modes": {}}
    pv = load_providers()
    out["providers"] = [{"slug": k, "label": v.get("label", k), "key_env": v.get("api_key_env"),
                         "key_set": bool(os.environ.get(v.get("api_key_env", ""), "")),
                         "authors": v.get("authors") or [], "direct_model": v.get("direct_model", "strip_author")}
                        for k, v in (pv.get("providers") or {}).items()]
    out["picker_authors"] = pv.get("picker_authors") or []
    for mode in ("demo", "live"):
        cfg_dir = _cfg_dir(mode)
        try:
            systems = load_systems(cfg_dir / "systems.yaml")
        except FileNotFoundError:
            continue
        ta = load_yaml(cfg_dir / "track_a.yaml")
        tb = load_yaml(cfg_dir / "track_b.yaml")
        ld = load_yaml(cfg_dir / "load.yaml")
        baseline = ta.get("baseline", BASELINE)

        def sysinfo(name):
            s = systems.get(name)
            if not s:
                return {"name": name, "configured": False, "available": False}
            return {"name": name, "configured": True, "available": True if mode == "demo" else s.available,
                    "key_env": s.api_key_env, "base_url": s.base_url, "model": s.models.get(ta.get("model_alias", "small")),
                    "router_model": s.router_model,
                    "todo": "TODO" in Path(cfg_dir / "systems.yaml").read_text().split(f"name: {name}")[-1].split("- name:")[0]}

        wls = []
        for w in tb.get("workloads", []):
            p = ROOT / w["path"]
            n = None
            if p.exists():
                n = sum(1 for line in open(p, encoding="utf-8") if line.strip())
            wls.append({"name": w["name"], "path": w["path"], "exists": p.exists(), "items": n})
        out["modes"][mode] = {
            "systems": [sysinfo(REF), sysinfo(RIVAL), sysinfo(baseline)],
            "pin_support": {n: bool(systems[n].pin) for n in (REF, RIVAL) if n in systems},
            "baseline": baseline,
            "shapes": ta.get("shapes", []),
            "workloads": wls,
            "model_baselines": tb.get("baselines", []),
            "model_baselines_ready": [b for b in tb.get("baselines", [])
                                      if b.split(":")[0] in systems and (mode == "demo" or systems[b.split(":")[0]].available)],
            "load": {"rates": ld.get("rates_rps", []), "step_duration_s": ld.get("step_duration_s", 60),
                     "slo_p99_ttft_ms": ld.get("slo_p99_ttft_ms")},
            "defaults": {"requests_per_cell": ta.get("requests_per_cell", 20) if mode == "demo" else 20,
                         "warmup": ta.get("warmup_requests", 3) if mode == "demo" else 5},
        }
    return out


@app.get("/api/models")
def list_models(refresh: bool = False):
    data, err = fetch_catalog(force=refresh)
    authors = load_providers().get("picker_authors") or []
    rows = []
    for m in data:
        mid = m.get("id", "")
        author = mid.split("/", 1)[0]
        if author not in authors or mid.endswith(":free"):
            continue
        pr = m.get("pricing") or {}
        rows.append({"id": mid, "name": m.get("name") or mid, "author": author, "created": m.get("created") or 0,
                     "prompt_per_m": _per_m(pr.get("prompt")), "completion_per_m": _per_m(pr.get("completion")),
                     "context": m.get("context_length")})
    rows.sort(key=lambda r: (authors.index(r["author"]), -r["created"]))
    return {"models": rows, "error": err}


_REF_MODELS: dict[str, Any] = {"t": 0.0, "models": None, "error": None}


def _norm(x: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def fetch_ref_models(force: bool = False) -> tuple[list[dict], str | None]:
    """Concentrate's GET /models: [{id, aliases, author, price {input, output} (cheapest provider),
    providers, temperature}]. Cached for an hour."""
    if not force and _REF_MODELS["models"] is not None and time.time() - _REF_MODELS["t"] < 3600:
        return _REF_MODELS["models"], None
    import httpx
    try:
        s_ = load_systems(_cfg_dir("live") / "systems.yaml")[REF]
        if not s_.available:
            return [], f"{s_.api_key_env} not set"
        from ..client import build_headers
        r = httpx.get(f"{s_.base_url}/models", headers=build_headers(s_), timeout=20)
        r.raise_for_status()
        body = r.json()
        items = body.get("data", body) if isinstance(body, dict) else body
    except Exception as e:  # noqa: BLE001
        return (_REF_MODELS["models"] or []), f"Concentrate model list unavailable: {e}"
    out = []
    for x in items or []:
        if not isinstance(x, dict):
            out.append({"id": str(x), "aliases": [], "author": None, "price": None, "providers": [], "temperature": None})
            continue
        prices, temps = [], []
        for pv in (x.get("providers") or {}).values():
            try:
                t = pv["pricing"][0]["tokens"]
                prices.append((float(t["input"]["price"]["USD"]) / float(t["input"].get("units", 1e6)) * 1e6,
                               float(t["output"]["price"]["USD"]) / float(t["output"].get("units", 1e6)) * 1e6))
            except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError):
                pass
            sup = (pv.get("supports") or {}).get("temperature")
            if sup is not None:
                temps.append(bool(sup))
        cheapest = min(prices, key=lambda p: p[0] + p[1]) if prices else None
        out.append({"id": x.get("id") or x.get("slug"), "aliases": x.get("aliases") or [],
                    "author": (x.get("author") or {}).get("slug") if isinstance(x.get("author"), dict) else x.get("author"),
                    "price": {"input": cheapest[0], "output": cheapest[1]} if cheapest else None,
                    "providers": list((x.get("providers") or {}).keys()),
                    "temperature": (any(temps) if temps else None)})
    _REF_MODELS.update(t=time.time(), models=out, error=None)
    return out, None


def match_ref_model(model: str, ref_models: list[dict]) -> dict | None:
    """Map an OpenRouter-style id ("anthropic/claude-opus-5.5") to Concentrate's model entry by id,
    then alias, then a punctuation-insensitive comparison (5.5 == 5-5)."""
    if not model:
        return None
    tail = model.split("/", 1)[-1]
    for want in (model, tail):
        for m in ref_models:
            if want == m["id"] or want in m["aliases"]:
                return m
    for want in (_norm(model), _norm(tail)):
        for m in ref_models:
            if want in {_norm(m["id"]), *(_norm(a) for a in m["aliases"])}:
                return m
    return None


@app.get("/api/ref-models")
def ref_models(refresh: bool = False):
    models, err = fetch_ref_models(force=refresh)
    return {"ids": [m["id"] for m in models], "models": models, "error": err}


@app.get("/api/models/{author}/{slug}/providers")
def model_providers(author: str, slug: str):
    import httpx
    try:
        r = httpx.get(f"{OPENROUTER_API}/models/{author}/{slug}/endpoints", timeout=20)
        r.raise_for_status()
        eps = (r.json().get("data") or {}).get("endpoints") or []
    except Exception as e:  # noqa: BLE001
        return {"providers": [], "error": f"Couldn't load providers for {author}/{slug}: {e}"}
    known = load_providers().get("providers") or {}
    out: dict[str, dict] = {}
    for ep in eps:
        tag = (ep.get("tag") or ep.get("provider_name") or "").split("/")[0].lower()
        if not tag or tag in out:
            continue
        pr = ep.get("pricing") or {}
        pv = known.get(tag)
        out[tag] = {"slug": tag, "name": ep.get("provider_name") or tag,
                    "prompt_per_m": _per_m(pr.get("prompt")), "completion_per_m": _per_m(pr.get("completion")),
                    "direct": bool(pv), "key_set": bool(pv and os.environ.get(pv.get("api_key_env", ""), "")),
                    "direct_model": pv.get("direct_model") if pv else None}
    order = list(known)
    rows = sorted(out.values(), key=lambda x: (order.index(x["slug"]) if x["slug"] in order else 99, x["name"]))
    return {"providers": rows, "error": None}


@app.post("/api/runs")
async def create_run(opts: RunOptions):
    if not (opts.track_a or opts.track_b or opts.load):
        raise HTTPException(400, "Pick at least one test.")
    active = [r for r in RUNS.values() if r.status in ("queued", "running")]
    if active:
        raise HTTPException(409, f"Run {active[0].id} is still going. Wait for it or cancel it first.")
    run = Run(opts)
    RUNS[run.id] = run
    run.save()
    run.task = asyncio.create_task(execute(run))
    return run.state()


@app.get("/api/runs")
def list_runs():
    saved = {s["id"]: s for s in _load_saved_runs()}
    for r in RUNS.values():
        saved[r.id] = r.state()
    rows = sorted(saved.values(), key=lambda s: s.get("created") or 0, reverse=True)
    return [{k: s.get(k) for k in ("id", "status", "created", "started", "finished", "error", "options")} for s in rows]


def _get_state(run_id: str) -> dict:
    if run_id in RUNS:
        return RUNS[run_id].state()
    p = RUNS_DIR / run_id / "state.json"
    if not p.exists():
        raise HTTPException(404, "run not found")
    st = json.loads(p.read_text())
    if st.get("status") in ("running", "queued"):
        st["status"] = "interrupted"
    return st


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    return _get_state(run_id)


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str):
    run = RUNS.get(run_id)
    if not run or not run.task or run.task.done():
        raise HTTPException(409, "run is not active")
    run.task.cancel()
    return {"ok": True}


@app.get("/api/runs/{run_id}/results")
def get_results(run_id: str):
    st = _get_state(run_id)
    mode = (st.get("options") or {}).get("mode", "live")
    cfg_dir = _cfg_dir(mode)
    run_dir = RUNS_DIR / run_id
    load_cfg = load_yaml(run_dir / "cfg_load.yaml") if (run_dir / "cfg_load.yaml").exists() else {}
    prices = run_dir / "prices.yaml" if (run_dir / "prices.yaml").exists() else (
        cfg_dir / "prices.yaml" if (cfg_dir / "prices.yaml").exists() else ROOT / "configs/prices.yaml")
    catalog = {}
    if mode == "live":
        data, _ = fetch_catalog()
        for m in data:
            pr = m.get("pricing") or {}
            pi, po = _per_m(pr.get("prompt")), _per_m(pr.get("completion"))
            if pi is not None and po is not None and (pi or po):
                catalog[m["id"]] = {"input": pi, "output": po}
        refs, _ = fetch_ref_models()
        for m in refs:
            if m.get("price"):
                for key in (m["id"], *m["aliases"]):
                    catalog.setdefault(key, m["price"])
    summary = build_summary(run_dir, prices, REF, RIVAL, load_cfg, catalog=catalog)
    summary["mode"] = mode
    summary["models"] = st.get("models") or {}
    return JSONResponse(summary)


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn
    print(f"gateway-bench UI -> http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
