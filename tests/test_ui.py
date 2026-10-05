import os
import time

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


def test_demo_run_end_to_end(tmp_path, monkeypatch):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.chdir(repo)
    from gateway_bench.ui import server

    monkeypatch.setattr(server, "RUNS_DIR", tmp_path / "ui")
    with TestClient(server.app) as c:
        cfg = c.get("/api/config").json()
        assert [s["name"] for s in cfg["modes"]["demo"]["systems"]][:2] == ["concentrate", "openrouter"]

        r = c.post("/api/runs", json={"mode": "demo", "requests_per_cell": 3, "warmup": 1,
                                      "shapes": ["tiny"], "max_items": 6, "load": False})
        assert r.status_code == 200, r.text
        run_id = r.json()["id"]

        for _ in range(120):
            st = c.get(f"/api/runs/{run_id}").json()
            if st["status"] not in ("queued", "running"):
                break
            time.sleep(0.25)
        assert st["status"] == "done", st

        res = c.get(f"/api/runs/{run_id}/results").json()
        keys = {m["key"] for m in res["metrics"]}
        assert {"ttft_p50", "accuracy", "cost_1k"} <= keys
        assert res["score"]["total"] == len(res["metrics"])
        assert any(row["label"] == "concentrate" for row in res["frontier"])
        assert run_id in [x["id"] for x in c.get("/api/runs").json()]


def test_rejects_empty_run():
    from gateway_bench.ui import server
    with TestClient(server.app) as c:
        r = c.post("/api/runs", json={"mode": "demo", "track_a": False, "track_b": False, "load": False})
        assert r.status_code == 400


def test_model_pin_and_direct_baseline(monkeypatch):
    from gateway_bench.config import load_systems
    from gateway_bench.ui import server

    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setenv("CONCENTRATE_API_KEY", "x")
    monkeypatch.setenv("NOVITA_API_KEY", "x")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    systems = load_systems("configs/systems.yaml")
    logs = []

    o = server.RunOptions(mode="live", model="deepseek/deepseek-chat", upstream="novita")
    alias, gw, base, info = server.resolve_model_setup(o, systems, "live", "small", "direct-openai", logs.append)
    orr = next(s for s in gw if s.name == "openrouter")
    assert orr.models[alias] == "deepseek/deepseek-chat"
    assert orr.extra_body["provider"] == {"only": ["novita"], "allow_fallbacks": False}
    assert "provider" not in systems["openrouter"].extra_body          # Track B systems untouched
    assert any("concentrate" in w for w in info["warnings"])            # no pin configured yet
    assert base.name == "direct-novita" and base.models[alias] == "deepseek/deepseek-chat"  # Novita keeps author/
    assert base.base_url == "https://api.novita.ai/openai/v1"

    # pinned to DeepSeek itself: id loses the author prefix, but no key -> baseline skipped
    o2 = server.RunOptions(mode="live", model="deepseek/deepseek-chat", upstream="deepseek")
    _, _, base2, info2 = server.resolve_model_setup(o2, systems, "live", "small", "direct-openai", logs.append)
    assert base2 is None and info2["baseline"] is None

    # explicit override of the direct id
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    o3 = server.RunOptions(mode="live", model="deepseek/deepseek-chat", upstream="deepseek", direct_model="deepseek-flash")
    _, _, base3, _ = server.resolve_model_setup(o3, systems, "live", "small", "direct-openai", logs.append)
    assert base3.models["__run__"] == "deepseek-flash"


def test_models_endpoint_filters_catalog(monkeypatch):
    import httpx
    from gateway_bench.ui import server

    catalog = {"data": [
        {"id": "z-ai/glm-x", "name": "GLM X", "created": 2, "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
        {"id": "openai/gpt-x:free", "name": "free", "created": 3, "pricing": {}},
        {"id": "meta-llama/llama-x", "name": "Llama", "created": 4, "pricing": {}},
        {"id": "openai/gpt-x", "name": "GPT X", "created": 1, "pricing": {"prompt": "0.000002", "completion": "0.000008"}},
    ]}
    monkeypatch.setattr(httpx, "get", lambda url, **kw: httpx.Response(200, json=catalog, request=httpx.Request("GET", url)))
    server._CATALOG.update(t=0.0, data=None)
    with TestClient(server.app) as c:
        r = c.get("/api/models").json()
    assert [m["id"] for m in r["models"]] == ["openai/gpt-x", "z-ai/glm-x"]      # picker order, no :free, no others
    assert abs(r["models"][0]["prompt_per_m"] - 2.0) < 1e-9
    server._CATALOG.update(t=0.0, data=None)
