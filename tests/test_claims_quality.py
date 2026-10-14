import json
import os
import time

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


def test_find_published_scores_walks_unknown_shapes():
    from gateway_bench.ui.claims import find_published_scores, snapshot_scores
    body = {"data": [
        {"benchmark": "GPQA Diamond", "results": [
            {"model": "anthropic/claude-opus-5.5", "accuracy": 0.902, "accuracy_stddev": 0.003, "avg_cost_per_task": 0.054},
            {"model": "openai/gpt-x", "accuracy": 0.5}]},
        {"benchmark_slug": "mmlu-pro", "model_slug": "claude-opus-5.5", "accuracy": 88.0},
    ]}
    got = find_published_scores(body, "anthropic/claude-opus-5.5")
    assert abs(got["gpqa_diamond"]["accuracy"] - 90.2) < 1e-9 and abs(got["gpqa_diamond"]["stddev"] - 0.3) < 1e-9
    assert got["mmlu_pro"]["accuracy"] == 88.0
    assert find_published_scores(body, "openai/other") == {}
    assert snapshot_scores("x/y", "Anthropic: Claude Opus 5.5")["gpqa_diamond"]["accuracy"] == 90.2


def test_build_claims_verdicts():
    from gateway_bench.ui.claims import build_claims
    ref = [{"id": "claude-opus-5-5", "aliases": ["opus-5.5"], "author": "anthropic", "price": {"input": 4, "output": 20},
            "providers": ["anthropic", "vertex"]},
           {"id": "glm-5.3", "aliases": [], "author": "z-ai", "price": {"input": 1, "output": 4}, "providers": ["z-ai"]}]
    orm = [{"id": "anthropic/claude-opus-5.5", "name": "Anthropic: Claude Opus 5.5", "pricing": {"prompt": "0.000004", "completion": "0.00002"}},
           {"id": "z-ai/glm-5.3", "pricing": {"prompt": "0.0000009", "completion": "0.000004"}},
           {"id": "z-ai/glm-5.3:free", "pricing": {"prompt": "0", "completion": "0"}}]

    def match(i):
        from gateway_bench.ui.server import match_ref_model
        return match_ref_model(i, ref)
    quality = {"model": "anthropic/claude-opus-5.5", "sets": [
        {"set": "gpqa_diamond", "n": 198, "acc": {"concentrate": 89.4, "openrouter": 90.9},
         "published": {"accuracy": 90.2, "stddev": 0.3, "source": "test"}}]}
    speed = {"run_id": "r1", "model": "anthropic/claude-opus-5.5", "upstream": "vertex", "or_ttft_p50": 2900.0,
             "or_tps_p50": 80.0, "ref_ttft_p50": 2100.0, "ref_tps_p50": 72.0, "or_reported_vs_list": 1.0,
             "published_endpoints": [{"provider": "Google Vertex", "tag": "vertex", "ttft_p50_ms": 2790.0, "tps_p50": 85.0,
                                      "uptime_30m": 99.9, "quantization": None}]}
    out = build_claims({"ref_models": ref, "or_models": orm, "match": match,
                        "speed": speed, "quality": quality})
    v = {c["id"]: c["verdict"] for c in out["claims"]}
    assert not {"conc_catalog", "or_models", "or_providers"} & set(v)   # catalog-size claims are out of scope
    assert v["conc_nofees"] == "partly"                 # glm is pricier on Concentrate in this fixture
    assert out["stats"]["or_ttft_p50"] == 2900.0 and out["stats"]["or_billed_ratio"] == 1.0
    assert v["or_nomarkup"] == "holds"
    assert v["or_latency"] == "holds"
    assert v["or_gpqa"] == "holds"
    assert v["conc_oneline"] == "partly"
    assert any(g["key"] == "gpqa_diamond" for g in out["glossary"])


def test_quality_summary_paired():
    import pandas as pd
    from gateway_bench.ui.summary import quality_summary
    rows = []
    for i in range(20):
        for lab, ok in (("concentrate", i % 5 != 0), ("openrouter", i % 4 != 0)):
            rows.append({"meta.label": lab, "system": lab, "meta.set": "gpqa_diamond", "meta.item_id": str(i),
                         "meta.rep": 0, "correct": ok, "ok": True, "cost_usd": 0.01, "e2e_ms": 1000.0,
                         "output_tokens": 300, "cost_source": "reported"})
    rows.append({"meta.label": "openrouter", "system": "openrouter", "meta.set": "gpqa_diamond", "meta.item_id": "x",
                 "meta.rep": 0, "correct": None, "ok": False, "cost_usd": None, "e2e_ms": 5.0, "output_tokens": None,
                 "cost_source": "unknown"})
    q = quality_summary(pd.DataFrame(rows), "concentrate", "openrouter",
                        {"model": "m", "benchmarks": {"gpqa_diamond": {"accuracy": 90.0}}})
    s = q["sets"][0]
    assert s["n"] == 20 and s["acc"]["concentrate"] == 80.0 and s["acc"]["openrouter"] == 75.0
    assert s["both_right"] + s["only_ref"] + s["only_rival"] + s["both_wrong"] == 20
    assert s["failed"]["openrouter"] == 1 and s["published"]["accuracy"] == 90.0
    assert 0 < s["p"] <= 1


def test_demo_quality_run(tmp_path, monkeypatch):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.chdir(repo)
    from gateway_bench.ui import server
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path / "ui")
    with TestClient(server.app) as c:
        sets = c.get("/api/quality/sets?mode=demo").json()
        assert sets["sets"][0]["ready"]
        rid = c.post("/api/runs", json={"mode": "demo", "track_a": False, "track_b": False, "quality": True,
                                        "quality_items": 6}).json()["id"]
        for _ in range(200):
            st = c.get(f"/api/runs/{rid}").json()
            if st["status"] not in ("queued", "running"):
                break
            time.sleep(0.05)
        assert st["status"] in ("done", "done_with_errors"), st
        res = c.get(f"/api/runs/{rid}/results").json()
        assert res["quality"]["sets"][0]["n"] > 0
        assert "accuracy_q" in {m["key"] for m in res["metrics"]}
        assert json.loads((tmp_path / "ui" / rid / "track_q.jsonl").read_text().splitlines()[0])["track"] == "Q"
