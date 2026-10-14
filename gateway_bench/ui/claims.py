"""Claims check: what Concentrate and OpenRouter say on their websites, and what we can measure.

Every claim has: who said it, where, what it means in plain words, how we check it, what we found and
a verdict. Verdicts: "holds", "partly", "doesn't hold", "not tested" (with the reason), "info".
The live numbers come from each gateway's own API and from your saved runs, so the page updates
whenever you run something new.
"""
from __future__ import annotations

import math
import re
from typing import Any

CONC_URL = "https://concentrate.ai"
OR_URL = "https://openrouter.ai"
OR_FAQ = "https://openrouter.ai/docs/faq"
OR_BENCH = "https://openrouter.ai/benchmarks"
OR_GPQA = "https://openrouter.ai/benchmarks/gpqa-diamond"
OR_ROUTERS = "https://openrouter.ai/benchmarks/routers"

# What OpenRouter's GPQA Diamond leaderboard showed when we read it (Oct 2026). Used only when the
# Benchmarks API can't be reached; the live API is preferred and is labelled as such.
GPQA_SNAPSHOT = [
    {"name": "Claude Opus 5.5", "ids": ["anthropic/claude-opus-5.5"], "accuracy": 90.2, "stddev": 0.3, "cost_per_task": 0.054},
    {"name": "GPT-6 Luna", "ids": [], "accuracy": 87.4, "stddev": None, "cost_per_task": 0.003},
    {"name": "DeepSeek V4.1 Flash", "ids": [], "accuracy": 88.8, "stddev": 0.9, "cost_per_task": 0.025},
    {"name": "GLM 5.3 Flash", "ids": [], "accuracy": 85.0, "stddev": None, "cost_per_task": None},
]

# Plain-language guide to the benchmarks quoted on both sites.
GLOSSARY = [
    {"key": "swe_bench", "name": "SWE-bench (Verified / Multimodal / Pro)", "kind": "model", "measures":
     "Real GitHub issues from open-source projects. The model gets the repo and the issue and must write a patch; "
     "the project's own tests decide pass or fail. Verified is a 500-task subset checked by humans; Multimodal adds "
     "issues with screenshots (front-end bugs).",
     "reads_as": "70% = the model fixed 70 of every 100 real bugs well enough that the hidden tests pass.",
     "gateway": "Needs an agent harness (repo checkout, tools, many turns) and hours per run. The score belongs to "
                "model + harness. A gateway can't raise it; it can only lower it (wrong provider, cut-off output)."},
    {"key": "gpqa_diamond", "name": "GPQA Diamond", "kind": "model", "measures":
     "198 graduate-level biology, chemistry and physics questions written by PhDs to be 'Google-proof'. "
     "Four options each. Experts in the field get about 65%; skilled non-experts with web access about 34%; "
     "random guessing 25%.",
     "reads_as": "90% = 178 of the 198 questions right. OpenRouter reruns it and averages over 90 days, so "
                 "± is the spread between reruns.",
     "gateway": "Auto-graded and cheap, so we rerun it through both gateways on the same model (Quality parity)."},
    {"key": "mmlu_pro", "name": "MMLU-Pro", "kind": "model", "measures":
     "12,000 harder multiple-choice questions across 14 subjects, 10 options each (random guess = 10%).",
     "reads_as": "Broad knowledge and reasoning. Top models score in the 80s.",
     "gateway": "Rerun in Quality parity."},
    {"key": "gsm8k", "name": "GSM8K", "kind": "model", "measures":
     "1,319 grade-school math word problems with a single number as the answer.",
     "reads_as": "Frontier models are near the ceiling (95%+). Useful as a sanity check: a drop here means "
                 "something is broken, not that the model is weaker.",
     "gateway": "Rerun in Quality parity."},
    {"key": "aa_index", "name": "Artificial Analysis Intelligence Index", "kind": "model", "measures":
     "A composite score from Artificial Analysis that averages about ten evals (reasoning, knowledge, math, coding, "
     "agentic tasks) into one number.",
     "reads_as": "Higher = generally more capable. Only comparable within the same index version.",
     "gateway": "Third-party composite; not something a gateway changes or that we rerun."},
    {"key": "terminal_bench", "name": "Terminal-Bench", "kind": "model", "measures":
     "Tasks done in a real Linux terminal (build this, fix that config, train a small model) graded by checks "
     "on the final machine state.",
     "reads_as": "Share of tasks fully completed by an agent.",
     "gateway": "Agentic, needs a sandbox per task. Not a gateway test."},
    {"key": "tau", "name": "τ²-bench / Tau3 (e.g. Banking, Retail, Airline)", "kind": "model", "measures":
     "An agent talks to a simulated customer and uses tools (look up account, refund, rebook) under a policy. "
     "Graded on whether the final database state is right.",
     "reads_as": "Tool use + following rules over a long conversation.",
     "gateway": "Sensitive to how a gateway passes tool calls through. Worth adding later as a tool-calling parity test."},
    {"key": "mmmu", "name": "MMMU Pro (Vision)", "kind": "model", "measures":
     "College-level questions that need an image (charts, diagrams, chemistry structures).",
     "reads_as": "Multimodal reasoning.",
     "gateway": "Tests image passthrough as much as the model."},
    {"key": "deep_swe", "name": "Deep SWE / SWE Atlas QA", "kind": "model", "measures":
     "Long software-engineering tasks (Deep SWE) and questions about large codebases (SWE Atlas QA).",
     "reads_as": "Coding depth beyond single-issue fixes.",
     "gateway": "Agentic; not a gateway test."},
    {"key": "router_index", "name": "OpenRouter Router Index", "kind": "router", "measures":
     "OpenRouter's comparison of auto-routers (its Auto Router, Pareto, Fugu and others) on several of the "
     "benchmarks above: accuracy vs cost when the router chooses the model.",
     "reads_as": "This is the one that is about routing, not models. Concentrate isn't listed on it.",
     "gateway": "Our Routing quality test is the same idea on open question sets."},
]


def _norm(x: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def _bench_key(name: str) -> str:
    n = _norm(name)
    if "gpqa" in n:
        return "gpqa_diamond"
    if "mmlupro" in n:
        return "mmlu_pro"
    if "gsm8k" in n:
        return "gsm8k"
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _num(v) -> float | None:
    if isinstance(v, dict):
        for k in ("p50", "median", "value", "avg", "mean"):
            if k in v:
                return _num(v[k])
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def find_published_scores(body: Any, model_id: str, model_name: str | None = None) -> dict[str, dict]:
    """Pull this model's scores out of OpenRouter's Benchmarks API response.

    The response shape isn't fixed in our code: walk it, carry identifying strings (model, benchmark)
    down from parents, and keep every dict that has an accuracy for this model."""
    targets = {_norm(model_id), _norm(model_id.split("/", 1)[-1])}
    if model_name:
        targets.add(_norm(model_name.split(":", 1)[-1]))
    targets.discard("")
    found: dict[str, dict] = {}

    def walk(node, ctx):
        if isinstance(node, list):
            for x in node:
                walk(x, ctx)
            return
        if not isinstance(node, dict):
            return
        ctx = dict(ctx)
        for k, v in node.items():
            if isinstance(v, str):
                kl = k.lower()
                if "model" in kl or kl in ("slug", "permaslug", "id", "name", "canonical_slug"):
                    ctx.setdefault("model_vals", [])
                    ctx["model_vals"] = ctx["model_vals"] + [v]
                if "benchmark" in kl or kl in ("task", "eval", "eval_name", "benchmark_slug"):
                    ctx["bench"] = v
        acc = node.get("accuracy", node.get("score"))
        if acc is not None and _num(acc) is not None:
            vals = {_norm(v) for v in ctx.get("model_vals", [])}
            if vals & targets:
                a = _num(acc)
                sd = _num(node.get("accuracy_stddev", node.get("stddev")))
                scale = 100 if a <= 1 else 1
                key = _bench_key(ctx.get("bench") or "unknown")
                found.setdefault(key, {"accuracy": a * scale, "stddev": sd * scale if sd is not None else None,
                                       "cost_per_task": _num(node.get("avg_cost_per_task")),
                                       "tasks": _num(node.get("total_tasks")),
                                       "last_run": node.get("last_run_timestamp"),
                                       "source": "OpenRouter Benchmarks API (live)"})
        for v in node.values():
            if isinstance(v, (dict, list)):
                walk(v, ctx)

    walk(body, {})
    return found


def snapshot_scores(model_id: str, model_name: str | None) -> dict[str, dict]:
    nm = _norm((model_name or "").split(":", 1)[-1])
    for row in GPQA_SNAPSHOT:
        if model_id in row["ids"] or (nm and _norm(row["name"]) == nm):
            return {"gpqa_diamond": {"accuracy": row["accuracy"], "stddev": row["stddev"],
                                     "cost_per_task": row["cost_per_task"], "tasks": None, "last_run": None,
                                     "source": f"{OR_GPQA} (page snapshot, Oct 2026)"}}
    return {}


def parse_endpoints(eps: list[dict]) -> list[dict]:
    """OpenRouter's per-provider stats for one model (needs an API key for latency/throughput)."""
    rows = []
    for ep in eps or []:
        lat = _num(ep.get("latency_last_30m"))
        if lat is not None and lat < 60:        # seconds -> ms
            lat *= 1000
        up = _num(ep.get("uptime_last_30m"))
        rows.append({"provider": ep.get("provider_name") or ep.get("tag"),
                     "tag": (ep.get("tag") or ep.get("provider_name") or "").split("/")[0].lower(),
                     "ttft_p50_ms": lat, "tps_p50": _num(ep.get("throughput_last_30m")),
                     "uptime_30m": (up * 100 if up is not None and up <= 1 else up),
                     "quantization": ep.get("quantization")})
    return rows


def _verdict_close(ours: float | None, theirs: float | None, tol: float) -> str | None:
    if ours is None or theirs is None or not theirs:
        return None
    return "holds" if abs(ours - theirs) / abs(theirs) <= tol else "doesn't hold"


def fmt_ms(v):
    return "–" if v is None else (f"{v / 1000:.2f} s" if v >= 1000 else f"{v:.0f} ms")


def build_claims(ctx: dict) -> dict:
    ref_models = ctx.get("ref_models") or []
    or_models = ctx.get("or_models") or []
    match = ctx.get("match")                 # callable(or_id) -> ref entry | None
    speed = ctx.get("speed") or {}
    qual = ctx.get("quality") or {}
    claims: list[dict] = []

    def add(**kw):
        kw.setdefault("evidence", [])
        claims.append(kw)

    # ------------------------------------------------ Concentrate (API, price and integration claims only)
    # list price vs OpenRouter (= provider list price, since OpenRouter adds no per-request markup)
    cmp_rows = []
    if match and or_models:
        for m in or_models:
            pr = m.get("pricing") or {}
            try:
                oi, oo = float(pr.get("prompt")) * 1e6, float(pr.get("completion")) * 1e6
            except (TypeError, ValueError):
                continue
            if not (oi or oo) or ":" in m.get("id", ""):
                continue
            r = match(m["id"])
            if r and r.get("price"):
                ri, ro = r["price"]["input"], r["price"]["output"]
                cmp_rows.append((m["id"], (ri * 4 + ro) / 5, (oi * 4 + oo) / 5))
    if cmp_rows:
        cheaper = sum(1 for _, a, b in cmp_rows if a < b * 0.995)
        same = sum(1 for _, a, b in cmp_rows if abs(a - b) <= b * 0.005)
        dearer = [(i, a, b) for i, a, b in cmp_rows if a > b * 1.005]
        v = "holds" if len(dearer) <= 0.05 * len(cmp_rows) else "partly"
        found = (f"Of {len(cmp_rows)} models listed on both: Concentrate's cheapest list price is lower on {cheaper}, "
                 f"the same on {same}, higher on {len(dearer)}.")
        ev = [{"label": "Higher on Concentrate", "value": ", ".join(f"{i} (${a:.2f} vs ${b:.2f})" for i, a, b in sorted(dearer, key=lambda x: x[2] - x[1])[:6])}] if dearer else []
    else:
        v, found, ev = "not tested", "Needs both model lists (Concentrate key + OpenRouter catalog).", []
    bill = speed.get("billing") or {}
    if bill.get("ref_ratio") is not None:
        ev.append({"label": f"Billing check (run {speed.get('run_id')})",
                   "value": f"Concentrate charged {bill['ref_ratio']:.2f}× list price for the tokens it processed"
                            + (f" — {bill['note']}" if bill.get("note") else "")
                            + ". Only meaningful if the dashboard amount covers exactly that run."})
    add(id="conc_nofees", who="concentrate", claim="No fees. Zero markup: you pay no more than going direct.", source=CONC_URL,
        means="The per-token price through Concentrate should equal the provider's own list price, with nothing added.",
        how="1) Compare Concentrate's list prices with OpenRouter's (which pass provider list prices through) for every "
            "model on both. 2) Fairness checks: enter what the dashboard charged for one run and compare with list price "
            "for the tokens processed (Concentrate doesn't return per-request cost).",
        found=found, verdict=v, evidence=ev)

    # one-line switch / model names
    if match and or_models:
        ids = [m["id"] for m in or_models if ":" not in m.get("id", "")]
        hit_any = [i for i in ids if match(i)]
        exact = [i for i in ids if (lambda r: r and (i in (r["id"], *r["aliases"]) or i.split("/", 1)[-1] in (r["id"], *r["aliases"])))(match(i))]
        found = (f"The API is OpenAI-compatible (same request and response format; our runs use one client for both). "
                 f"But model names differ: {len(exact)} of {len(ids)} OpenRouter model IDs work as-is on Concentrate "
                 f"(ID or alias), {len(hit_any)} after ignoring punctuation (5.5 vs 5-5).")
        v = "partly"
        ev = [{"label": "Example", "value": "anthropic/claude-opus-5.5 → 404 on Concentrate; it needs claude-opus-5-5 "
                                           "(aliases include opus-5.5 but not claude-opus-5.5)."}]
    else:
        v, found, ev = "not tested", "Needs both model lists.", []
    add(id="conc_oneline", who="concentrate", claim="Integrate in 30 seconds: change one line", source=CONC_URL,
        means="Point your existing OpenAI-style code at Concentrate's URL and it works.",
        how="Same client sends identical requests to both gateways in every run; count how many OpenRouter model IDs "
            "Concentrate accepts unchanged.", found=found, verdict=v, evidence=ev)

    # Concentrate's claim about OpenRouter
    rr = speed.get("or_reported_vs_list")
    add(id="conc_about_or", who="concentrate", claim="OpenRouter adds about 3% for payment processing plus a 3% platform fee",
        source=CONC_URL,
        means="A claim about the competitor: roughly 6% on top of what you'd pay providers.",
        how="Read OpenRouter's fee docs, then check per-request billed cost (usage.cost) against list price in your runs.",
        found=("OpenRouter's docs: 5.5% fee when you buy credits by card ($0.80 minimum), 5% for crypto; no markup on "
               "per-request inference prices. "
               + (f"In run {speed.get('run_id')}, OpenRouter's per-request cost was {rr:.3f}× list price." if rr else "")
               + " So the total is about 5.5%, charged once at top-up, not per request."),
        verdict="partly", evidence=[{"label": "Source", "value": OR_FAQ},
                                    {"label": "Example", "value": "$30 on the OpenRouter dashboard ≈ $28.13 of usage × 1.055"}])

    add(id="conc_bench", who="concentrate", claim="No benchmark scores published", source=CONC_URL,
        means="—", how="Read concentrate.ai, /models and /features/routing.",
        found="Concentrate's site makes no benchmark claims (no SWE-bench, GPQA or router leaderboard). Its claims are "
              "about price and integration, checked above. Concentrate is also not on OpenRouter's Router Index.",
        verdict="info", evidence=[{"label": "Router Index", "value": OR_ROUTERS}])

    # ------------------------------------------------ OpenRouter
    add(id="or_nomarkup", who="openrouter", claim="No markup on inference: you pay provider prices", source=OR_FAQ,
        means="Per-request price equals the provider's list price; the fee is on buying credits instead.",
        how="Compare usage.cost that OpenRouter returns on every request with list price × tokens.",
        found=(f"Run {speed.get('run_id')}: billed per request = {rr:.3f}× list price." if rr else
               "Run a Speed & fees test on a live model to check."),
        verdict=(("holds" if abs(rr - 1) < 0.01 else "doesn't hold") if rr else "not tested"))

    pub = speed.get("published_endpoints") or []
    up = speed.get("upstream")
    row = next((r for r in pub if up and r["tag"] == up), None)
    if pub and speed.get("or_ttft_p50") is not None:
        target = row or {"provider": "best provider", "ttft_p50_ms": min((r["ttft_p50_ms"] for r in pub if r["ttft_p50_ms"]), default=None),
                         "tps_p50": max((r["tps_p50"] for r in pub if r["tps_p50"]), default=None)}
        v1 = _verdict_close(speed["or_ttft_p50"], target["ttft_p50_ms"], 0.25)
        v2 = _verdict_close(speed.get("or_tps_p50"), target["tps_p50"], 0.25)
        vs = [x for x in (v1, v2) if x]
        v = "holds" if vs and all(x == "holds" for x in vs) else ("partly" if "holds" in vs else ("doesn't hold" if vs else "not tested"))
        found = (f"{speed.get('model')} via {target['provider']}: OpenRouter shows {fmt_ms(target['ttft_p50_ms'])} p50 latency and "
                 f"{target['tps_p50'] or 0:.0f} tok/s; we measured {fmt_ms(speed['or_ttft_p50'])} and "
                 f"{speed.get('or_tps_p50') or 0:.0f} tok/s through OpenRouter "
                 f"(Concentrate: {fmt_ms(speed.get('ref_ttft_p50'))}, {speed.get('ref_tps_p50') or 0:.0f} tok/s). "
                 "Within 25% counts as matching; their numbers are the last 30 minutes across all users and prompt sizes.")
        ev = [{"label": r["provider"], "value": f"{fmt_ms(r['ttft_p50_ms'])} · {r['tps_p50'] or 0:.0f} tok/s · uptime {r['uptime_30m'] or 0:.1f}%"
               + (f" · {r['quantization']}" if r.get("quantization") else "")} for r in pub[:8]]
    else:
        v, ev = "not tested", []
        found = ("Run Speed & fees on a live model: the run records OpenRouter's published per-provider latency, "
                 "throughput and uptime at start, then compares with what we measure.")
    add(id="or_latency", who="openrouter", claim="Per-provider latency, throughput and uptime (model pages)", source=OR_URL + "/models",
        means="OpenRouter shows live stats per provider so you can pick a fast one. Latency there = time to first token.",
        how="Snapshot GET /api/v1/models/{model}/endpoints at run start; compare with our measured streaming TTFT p50 and tok/s.",
        found=found, verdict=v, evidence=ev)

    qsets = {s_["set"]: s_ for s_ in (qual.get("sets") or [])}
    g = qsets.get("gpqa_diamond")
    pubg = (g or {}).get("published") or (qual.get("published") or {}).get("gpqa_diamond")
    if g and g.get("n") and pubg:
        ours = g["acc"]["openrouter"]
        se = math.sqrt(max(ours * (100 - ours), 1) / g["n"])
        tol = max(3.0, 2 * se + (pubg.get("stddev") or 0))
        v = "holds" if abs(ours - pubg["accuracy"]) <= tol else "doesn't hold"
        found = (f"{qual.get('model')}: OpenRouter publishes {pubg['accuracy']:.1f}%"
                 + (f" ±{pubg['stddev']:.1f}" if pubg.get("stddev") else "")
                 + f"; our rerun through OpenRouter got {ours:.1f}% and through Concentrate {g['acc']['concentrate']:.1f}% "
                   f"on {g['n']} questions (allowed gap ±{tol:.1f} points for sample noise).")
    elif pubg:
        v = "not tested"
        found = (f"OpenRouter publishes {pubg['accuracy']:.1f}% for this model. Run Quality parity with GPQA Diamond to check it.")
    else:
        v, found = "not tested", "Run Quality parity with GPQA Diamond on a model OpenRouter has benchmarked."
    add(id="or_gpqa", who="openrouter", claim="GPQA Diamond scores per model (Benchmarks)", source=OR_GPQA,
        means="OpenRouter reruns GPQA Diamond through its own routing and publishes accuracy, ± spread and cost per question.",
        how="Same 198 questions, same model, through both gateways (Quality parity). Compare with the published number.",
        found=found, verdict=v,
        evidence=[{"label": "Published source", "value": pubg.get("source")}] if pubg else [])

    add(id="or_swe", who="openrouter", claim="SWE-bench Multimodal 70.2% and Intelligence Index (Opus 5.5 model page)", source=OR_URL + "/anthropic/claude-opus-5.5",
        means="Third-party model scores (Vals AI, Artificial Analysis) shown on the model page. They describe the model, "
              "not OpenRouter.",
        how="Not rerun: agentic benchmarks need a full harness and hours per run, and a gateway can only lower the score. "
            "Quality parity is the gateway-relevant version: if both gateways match on GPQA/MMLU-Pro/GSM8K for the "
            "same model, there's no reason to expect a SWE-bench gap.",
        found="Shown as context; see the glossary for what each benchmark measures.", verdict="not tested")

    counts: dict[str, int] = {}
    for c in claims:
        counts[c["verdict"]] = counts.get(c["verdict"], 0) + 1
    return {"claims": claims, "counts": counts, "glossary": GLOSSARY,
            "stats": {
                "ref_ttft_p50": speed.get("ref_ttft_p50"), "or_ttft_p50": speed.get("or_ttft_p50"),
                "ref_tps_p50": speed.get("ref_tps_p50"), "or_tps_p50": speed.get("or_tps_p50"),
                "ref_billed_ratio": (speed.get("billing") or {}).get("ref_ratio"),
                "or_billed_ratio": speed.get("or_reported_vs_list"),
                "model": speed.get("model"), "run_id": speed.get("run_id")}}
