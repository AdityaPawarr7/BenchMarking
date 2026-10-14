import datetime as dt
import json
import os
import time

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _wait(c, rid):
    for _ in range(400):
        st = c.get(f"/api/runs/{rid}").json()
        if st["status"] not in ("queued", "running"):
            return st
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_demo_volume_and_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    from gateway_bench.ui import server
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path / "ui")
    with TestClient(server.app) as c:
        rid = c.post("/api/runs", json={"mode": "demo", "track_a": False, "track_b": False, "volume": True,
                                        "volume_levels": [{"requests": 30, "concurrency": 5}, {"requests": 60, "concurrency": 20}],
                                        "cache_probe": True}).json()["id"]
        st = _wait(c, rid)
        assert st["status"] in ("done", "done_with_errors"), st
        res = c.get(f"/api/runs/{rid}/results").json()
    levels = {(r["level"], r["system"]) for r in res["volume"]}
    assert ("60@20", "concentrate") in levels and ("30@5", "openrouter") in levels
    big = [r for r in res["volume"] if r["level"] == "60@20"]
    assert all(r["n"] == 60 and r["rps"] > 0 for r in big)
    keys = {m["key"] for m in res["metrics"]}
    assert {"vol_success", "vol_ttft_p90", "vol_rps"} <= keys
    assert {r["system"] for r in res["cache"]} == {"concentrate", "openrouter"}
    assert all(r["calls"] == 4 for r in res["cache"])


def test_client_reads_provider_and_cache_fields():
    from gateway_bench.client import CallResult, _served_provider, _usage_details
    r = CallResult(request_id="x", system="openrouter", model_requested="m")
    _usage_details(r, {"prompt_tokens": 9000, "completion_tokens": 10, "cost": 0.01,
                       "prompt_tokens_details": {"cached_tokens": 8000}, "completion_tokens_details": {"reasoning_tokens": 7}})
    _served_provider(r, {"provider": "DeepInfra", "model": "deepseek/x"})
    assert (r.cached_tokens, r.reasoning_tokens, r.cost_reported, r.provider_served) == (8000, 7, 0.01, "DeepInfra")
    c = CallResult(request_id="y", system="concentrate", model_requested="m", model_served="runware/deepseek-v4-1-flash")
    _served_provider(c, {})
    assert c.provider_served == "runware"


def test_served_provider_price_used_for_concentrate_cost():
    import pandas as pd
    from gateway_bench.pricing import PriceBook
    from gateway_bench.ui.summary import fill_costs
    df = pd.DataFrame([{"system": "concentrate", "model_served": "runware/ds", "model_requested": "ds", "input_tokens": 1e6,
                        "output_tokens": 0, "cost_reported": None},
                       {"system": "openrouter", "model_served": "deepseek/ds", "model_requested": "deepseek/ds", "input_tokens": 1e6,
                        "output_tokens": 0, "cost_reported": 0.02}])
    cat = {"concentrate:runware/ds": {"input": 0.15, "output": 0.6}, "concentrate:deepseek/ds": {"input": 0.3, "output": 1.2}}
    out = fill_costs(df, PriceBook({}), cat)
    assert list(out.cost_usd) == [0.15, 0.02]
    assert list(out.cost_source) == ["served_provider_list", "reported"]


def test_sweep_plan_slots_and_budget(monkeypatch):
    monkeypatch.chdir(REPO)
    from gateway_bench import sweep
    cfg = sweep.load_yaml("configs/top10.yaml")
    assert len(cfg["models"]) == 10
    refs = [{"id": m["concentrate"], "aliases": [], "price": {"input": 0.2, "output": 0.8},
             "provider_prices": {"a": {"input": 0.2, "output": 0.8}, "b": {"input": 0.4, "output": 1.6}}} for m in cfg["models"]]
    refs[6]["price"] = {"input": 4, "output": 20}          # Opus: over the 10k level's price cap
    models = sweep.resolve_models(cfg, refs, [], lambda *_: None)
    wed = dt.datetime(2026, 10, 14, 9)                       # Wednesday morning
    sun = dt.datetime(2026, 10, 18, 20)                      # Sunday evening = the weekly slot
    assert sweep.current_slot(wed) == "morning" and sweep.current_slot(sun) == "evening"
    u_morning = sweep.plan(cfg, "morning", wed, models)
    assert len(u_morning) == 20 and all("quality" not in u and "cache" not in u for u in u_morning)
    assert all([lv["requests"] for lv in u["volume"]["levels"]] == [100, 1000] for u in u_morning)
    u_week = sweep.plan(cfg, "evening", sun, models)
    opus = [u for u in u_week if u["model"]["name"] == "Claude Opus 5.5"]
    assert all(10000 not in [lv["requests"] for lv in u["volume"]["levels"]] for u in opus)
    assert any("quality" in u for u in u_week) and not any("quality" in u for u in u_week if u["mode"] == "default")
    trimmed = sweep.apply_budget(u_week, 1.0, lambda *_: None)
    assert sum(u["est_usd"] for u in trimmed) <= 1.0


def test_log_rows_and_sweeps_endpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    from gateway_bench import sweep
    from gateway_bench.ui import server
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path / "ui")
    monkeypatch.setattr(server, "build_catalog", lambda mode="live": {})
    rid = "r1"
    d = tmp_path / "ui" / rid
    d.mkdir(parents=True)
    rows = []
    for sys_ in ("concentrate", "openrouter"):
        for i in range(4):
            rows.append({"track": "A", "system": sys_, "model_requested": "m", "model_served": "p/m", "stream": True, "ok": True,
                         "ttft_ms": 100.0 + i + (50 if sys_ == "openrouter" else 0), "e2e_ms": 900.0, "first_text_ms": 120.0,
                         "input_tokens": 1300, "output_tokens": 500, "output_tps": 80.0, "cost_reported": 0.001,
                         "provider_served": "novita", "t_start": 1.0, "meta": {"shape": "short", "warmup": False, "rep": i}})
        for i in range(10):
            rows.append({"track": "V", "system": sys_, "model_requested": "m", "stream": True, "ok": i != 0, "status": 200 if i else 429,
                         "ttft_ms": 200.0, "e2e_ms": 1000.0, "input_tokens": 160, "output_tokens": 50, "cost_reported": 0.0001,
                         "t_start": 1.0 + i * 0.1, "meta": {"level": "10@5", "requests": 10, "concurrency": 5}})
    (d / "track_a.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    u = {"model": {"name": "GLM 5.3"}, "mode": "default"}
    out = sweep.log_rows(rid, u, "morning", "s1", None, dt.datetime(2026, 10, 14, 9, 0))
    speed = [r for r in out if r["test"] == "speed"]
    vol = [r for r in out if r["test"] == "volume"]
    assert len(speed) == 2 and speed[0]["ttft_p50_ms"] is not None and speed[0]["providers_served"] == "novita:4"
    assert len(vol) == 2 and vol[0]["success_pct"] == 90.0 and vol[0]["rps"] > 0
    csv_path = tmp_path / "sweep_log.csv"
    sweep.append_csv(out, csv_path)
    monkeypatch.setattr(sweep, "LOG_CSV", csv_path)
    monkeypatch.setattr(server, "ROOT", tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "top10.yaml").write_text("models: [{name: GLM 5.3}]\n")
    with TestClient(server.app) as c:
        res = c.get("/api/sweeps?days=100000").json()
    assert res["rows"][0]["model"] == "GLM 5.3"
    assert res["rows"][0]["concentrate"]["ttft_p50"] < res["rows"][0]["openrouter"]["ttft_p50"]
    assert res["rows"][0]["vol_level"] == "10@5"


def test_plist_escapes_paths():
    from pathlib import Path
    from gateway_bench.sweep import plist_text
    t = plist_text({"morning": "08:47"}, Path("/Users/a/Adi's Mac & Co/Bench"), "/x/gbench", Path("/tmp/l.log"))
    assert "Adi's Mac &amp; Co" in t and "<integer>8</integer>" in t and "<integer>47</integer>" in t
