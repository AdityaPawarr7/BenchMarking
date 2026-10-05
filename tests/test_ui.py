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
