"""Local web UI: configure and launch Concentrate-vs-OpenRouter runs, watch progress, view results.

Start with:  gbench ui            (http://127.0.0.1:8765)
API keys are read from .env on this machine and never sent to the browser.
"""
from __future__ import annotations

import asyncio
import copy
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
from .summary import build_summary, clean

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
                      "options": self.opts.model_dump(), "models": self.models, "logs": self.logs[-60:]})

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
    for s_ in gw:
        s_.models[alias] = o.model
        info["ids"][s_.name] = o.model
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
            cfg = {**ta_base, "baseline": base_sys.name if (o.include_baseline and base_sys) else None,
                   "model_alias": alias, "stream_modes": o.stream_modes or [True],
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
            wls = [w for w in base.get("workloads", []) if not o.workloads or w["name"] in o.workloads]
            cfg = {**base, "workloads": [{**w, "max_items": o.max_items} for w in wls], "routers": [REF, RIVAL],
                   "baselines": base.get("baselines", []) if o.include_model_baselines else []}
            path = _write_yaml(run.dir / "cfg_track_b.yaml", cfg)
            run.phases["Track B: routing quality"] = {"done": 0, "total": 0, "status": "running"}
            run.log("Track B started")
            await run_track_b(systems, path, run.dir, log=run.log, progress=run.progress("Track B: routing quality"),
                              prices_path=cfg_dir / "prices.yaml" if (cfg_dir / "prices.yaml").exists() else ROOT / "configs/prices.yaml",
                              out_path=run.dir / "track_b.jsonl")
            run.phases["Track B: routing quality"]["status"] = "done"
            run.save()

        if o.load:
            base = load_yaml(cfg_dir / "load.yaml")
            cfg = {**base, "rates_rps": o.rates or base["rates_rps"], "step_duration_s": o.step_duration_s,
                   "model_alias": alias}
            path = _write_yaml(run.dir / "cfg_load.yaml", cfg)
            run.phases["Load test"] = {"done": 0, "total": 0, "status": "running"}
            run.log("Load test started")
            await run_load(gw, path, run.dir, log=run.log,
                           progress=run.progress("Load test"), out_path=run.dir / "load.jsonl")
            run.phases["Load test"]["status"] = "done"

        run.status = "done"
        run.log("Run complete")
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
    summary = build_summary(run_dir, prices, REF, RIVAL, load_cfg)
    summary["mode"] = mode
    summary["models"] = st.get("models") or {}
    return JSONResponse(summary)


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn
    print(f"gateway-bench UI -> http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
