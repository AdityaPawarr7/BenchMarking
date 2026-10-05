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
        assert st["status"] in ("done", "done_with_errors"), st   # demo mocks fail ~1% on purpose

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


def test_preflight_stops_run_on_unknown_model(tmp_path, monkeypatch):
    """Reproduces the first live run: Concentrate rejects the model name -> stop before the test."""
    import sys
    import threading
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.chdir(repo)
    sys.path.insert(0, os.path.join(repo, "scripts"))
    from mock_gateway import serve
    from gateway_bench.ui import server

    strict = serve(9411, latency_ms=5, tok_ms=0, fail_rate=0, accuracy=1, price_per_mtok=1, name="concentrate",
                   known_models={"auto", "anthropic/claude-x-4-5"}, report_cost=False)
    loose = serve(9412, latency_ms=5, tok_ms=0, fail_rate=0, accuracy=1, price_per_mtok=1, name="openrouter")
    for s_ in (strict, loose):
        threading.Thread(target=s_.serve_forever, daemon=True).start()
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "systems.yaml").write_text(
        "systems:\n"
        "  - {name: concentrate, kind: router, base_url: 'http://127.0.0.1:9411/v1', api_key_env: T_KEY, tracks: [A, B], models: {small: x}, router_model: auto}\n"
        "  - {name: openrouter, kind: router, base_url: 'http://127.0.0.1:9412/v1', api_key_env: T_KEY, tracks: [A, B], models: {small: x}, router_model: openrouter/auto}\n")
    for f in ("track_a.yaml", "track_b.yaml", "load.yaml"):
        (cfg / f).write_text(open(os.path.join(repo, "configs/ui_demo", f)).read())
    monkeypatch.setenv("T_KEY", "k")
    monkeypatch.setattr(server, "_cfg_dir", lambda mode: cfg)
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path / "ui")

    def run(opts):
        with TestClient(server.app) as c:
            rid = c.post("/api/runs", json=opts).json()["id"]
            for _ in range(200):
                st = c.get(f"/api/runs/{rid}").json()
                if st["status"] not in ("queued", "running"):
                    return st, c.get(f"/api/runs/{rid}/results").json()
                time.sleep(0.05)
        raise AssertionError("run did not finish")

    base = {"mode": "live", "track_b": False, "requests_per_cell": 2, "warmup": 0, "shapes": ["tiny"],
            "include_baseline": False, "model": "anthropic/claude-x-4.5"}
    st, _ = run(base)
    assert st["status"] == "failed"
    assert "does not exist" in st["error"] and "concentrate" in st["error"]
    assert [c["ok"] for c in st["preflight"]] == [False, True]
    assert not (tmp_path / "ui" / st["id"] / "track_a.jsonl").exists()   # no requests wasted

    # same run with Concentrate's own name for the model -> passes, cost estimated from catalog
    server._CATALOG.update(t=time.time(), data=[{"id": "anthropic/claude-x-4.5", "pricing": {"prompt": "0.000003", "completion": "0.000015"}}])
    st, res = run({**base, "ref_model": "anthropic/claude-x-4-5"})
    server._CATALOG.update(t=0.0, data=None)
    assert st["status"] == "done", st
    assert st["models"]["ids"]["concentrate"] == "anthropic/claude-x-4-5"
    cost = next(m for m in res["metrics"] if m["key"] == "cost_mtok")
    assert cost["values"]["concentrate"] is not None and "estimated" in cost["note"]
    assert res["errors"] == []


def test_readable_error_and_results_with_failed_rows(tmp_path):
    import json
    from gateway_bench.ui.summary import build_summary, readable_error

    raw = json.dumps({"type": "error", "error": {"type": "invalid_request_error", "message": "This organization has been disabled."}})
    err = "status 400: " + json.dumps({"error": {"message": "Provider returned error", "code": 400, "metadata": {"raw": raw}}})
    assert readable_error(err) == "status 400: Provider returned error (upstream: This organization has been disabled.)"
    assert readable_error('status 404: {"error":{"message":"The model \'x\' does not exist"}}') == "status 404: The model 'x' does not exist"
    # stored errors used to be cut at 500 chars, which breaks the JSON; still recover the messages
    cut = err[:230]
    assert readable_error(cut).startswith("status 400: Provider returned error (upstream: This organization has been disabled")

    # every request failed (like the first live run): results must still build, with an Errors list
    rows = []
    for sys_ in ("concentrate", "openrouter"):
        for i in range(3):
            rows.append({"track": "A", "system": sys_, "model_requested": "m", "model_served": None, "stream": True,
                         "status": 404, "ok": False, "error": err, "ttft_ms": None, "e2e_ms": 5.0, "input_tokens": None,
                         "output_tokens": None, "output_tps": None, "cost_reported": None, "baseline": None,
                         "meta": {"shape": "tiny", "warmup": False, "rep": i}})
    (tmp_path / "track_a.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (tmp_path / "prices.yaml").write_text("models: {}\n")
    out = build_summary(tmp_path, tmp_path / "prices.yaml")
    assert {e["system"] for e in out["errors"]} == {"concentrate", "openrouter"}
    assert "organization has been disabled" in out["errors"][0]["top_errors"][0]["message"]
    succ = next(m for m in out["metrics"] if m["key"] == "success_a")
    assert succ["winner"] is None


def test_match_ref_model_uses_ids_aliases_and_punctuation():
    from gateway_bench.ui.server import match_ref_model
    refs = [
        {"id": "claude-opus-5-5", "aliases": ["claude-opus5.5", "opus-5.5", "opus-5-5"]},
        {"id": "gpt-5.5", "aliases": ["gpt-5-5", "gpt55"]},
        {"id": "glm-5.3", "aliases": ["glm5.3", "zhipu-glm-5.3"]},
    ]
    assert match_ref_model("anthropic/claude-opus-5.5", refs)["id"] == "claude-opus-5-5"   # 5.5 vs 5-5
    assert match_ref_model("opus-5.5", refs)["id"] == "claude-opus-5-5"                    # alias
    assert match_ref_model("openai/gpt-5-5", refs)["id"] == "gpt-5.5"
    assert match_ref_model("z-ai/glm-5.3-prime", refs) is None                             # different SKU: no guess


def test_temperature_omitted_when_none():
    import asyncio
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from gateway_bench.client import chat, make_client
    from gateway_bench.config import System

    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            data = json.dumps({"model": "m", "choices": [{"message": {"content": "OK"}}], "usage": {}}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", 9431), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    s = System(name="x", kind="gateway", base_url="http://127.0.0.1:9431/v1", api_key_env="NONE")

    async def go(t):
        async with make_client() as c:
            return await chat(c, s, "m", [{"role": "user", "content": "hi"}], stream=False, temperature=t)
    asyncio.run(go(None)); asyncio.run(go(0.0))
    srv.shutdown()
    assert "temperature" not in seen[0] and seen[1]["temperature"] == 0.0
