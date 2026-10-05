"""Local web UI: configure and launch Concentrate-vs-OpenRouter runs, watch progress, view results.

Start with:  gbench ui            (http://127.0.0.1:8765)
API keys are read from .env on this machine and never sent to the browser.
"""
from __future__ import annotations

import asyncio
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

from ..config import load_systems, load_yaml
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
                      "options": self.opts.model_dump(), "logs": self.logs[-60:]})

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

        if o.track_a:
            base = load_yaml(cfg_dir / "track_a.yaml")
            names = [REF, RIVAL]
            baseline = base.get("baseline", BASELINE)
            if o.include_baseline and baseline in systems and systems[baseline].available:
                names.append(baseline)
            elif o.include_baseline:
                run.log(f"[skip] baseline {baseline}: no API key, overhead vs direct won't be shown")
            cfg = {**base, "baseline": baseline if baseline in names else None,
                   "stream_modes": o.stream_modes or [True], "requests_per_cell": o.requests_per_cell,
                   "warmup_requests": o.warmup}
            if o.shapes:
                cfg["shapes"] = [s for s in base["shapes"] if s["name"] in o.shapes]
            path = _write_yaml(run.dir / "cfg_track_a.yaml", cfg)
            run.phases["Track A: speed & fees"] = {"done": 0, "total": 0, "status": "running"}
            run.log("Track A started")
            await run_track_a([systems[n] for n in names], path, run.dir, log=run.log,
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
            cfg = {**base, "rates_rps": o.rates or base["rates_rps"], "step_duration_s": o.step_duration_s}
            path = _write_yaml(run.dir / "cfg_load.yaml", cfg)
            run.phases["Load test"] = {"done": 0, "total": 0, "status": "running"}
            run.log("Load test started")
            await run_load([systems[REF], systems[RIVAL]], path, run.dir, log=run.log,
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


# ------------------------------------------------------------------ API

app = FastAPI(title="gateway-bench UI", docs_url="/api/docs")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def get_config():
    out: dict[str, Any] = {"ref": REF, "rival": RIVAL, "modes": {}}
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
    prices = cfg_dir / "prices.yaml" if (cfg_dir / "prices.yaml").exists() else ROOT / "configs/prices.yaml"
    summary = build_summary(run_dir, prices, REF, RIVAL, load_cfg)
    summary["mode"] = mode
    return JSONResponse(summary)


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn
    print(f"gateway-bench UI -> http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
