"""Build the JSON the UI renders: head-to-head metrics, chart series and tables for one run."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import analysis as A
from ..pricing import PriceBook
from ..results import read_results

_META_COLS = ("meta.warmup", "meta.shape", "meta.workload", "meta.label", "meta.role", "meta.item_id",
              "meta.target_rps", "baseline", "correct", "cost_usd", "model_served", "output_tps")


def clean(obj: Any) -> Any:
    """Make numpy/pandas values JSON-safe (NaN/inf -> None)."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if obj is pd.NaT:
        return None
    return obj


def _records(df: pd.DataFrame) -> list[dict]:
    return [] if df is None or df.empty else clean(df.to_dict("records"))


def _metric(key, label, unit, better, track, ref, rival, a_val, b_val, ci=None, p=None, note=None):
    m = {"key": key, "label": label, "unit": unit, "better": better, "track": track,
         "values": {ref: a_val, rival: b_val}, "winner": None, "delta_pct": None,
         "ci": ci, "p": p, "significant": None, "note": note}
    if a_val is None or b_val is None or (isinstance(a_val, float) and math.isnan(a_val)) \
            or (isinstance(b_val, float) and math.isnan(b_val)):
        return clean(m)
    if a_val == b_val:
        m["winner"] = "tie"
    else:
        ref_better = (a_val < b_val) if better == "lower" else (a_val > b_val)
        m["winner"] = ref if ref_better else rival
    if b_val:
        m["delta_pct"] = (a_val - b_val) / abs(b_val) * 100
    if ci is not None and ci[0] is not None and ci[1] is not None and not any(map(math.isnan, ci)):
        m["significant"] = bool(ci[0] > 0 or ci[1] < 0)
    elif p is not None and not math.isnan(p):
        m["significant"] = bool(p < 0.05)
    return clean(m)


def _cdf(values: pd.Series, points: int = 120) -> list[list[float]]:
    v = np.sort(pd.Series(values).dropna().to_numpy(dtype=float))
    if len(v) == 0:
        return []
    qs = np.linspace(0, 1, min(points, len(v)))
    xs = np.quantile(v, qs)
    return [[float(x), float(q)] for x, q in zip(xs, qs)]


def build_summary(run_dir: str | Path, prices_path: str | Path, ref: str = "concentrate",
                  rival: str = "openrouter", load_cfg: dict | None = None) -> dict:
    run_dir = Path(run_dir)
    files = sorted(run_dir.glob("*.jsonl"))
    if not files:
        return {"empty": True}
    df = read_results(files)
    if df.empty:
        return {"empty": True}
    for c in _META_COLS:
        if c not in df:
            df[c] = np.nan
    prices = PriceBook.load(prices_path)
    load_cfg = load_cfg or {}
    out: dict[str, Any] = {"empty": False, "ref": ref, "rival": rival, "metrics": []}
    metrics = out["metrics"]

    # ---------------- Track A
    ta = df[(df.track == "A") & (df["meta.warmup"] != True)].copy()  # noqa: E712
    if not ta.empty:
        a_tab = A.track_a_summary(df, prices)
        out["track_a"] = _records(a_tab)
        ta = A.add_costs(ta, prices)
        ok = ta[ta.ok == True]  # noqa: E712
        r_ok, v_ok = ok[ok.system == ref], ok[ok.system == rival]
        if len(r_ok) and len(v_ok):
            for q, tag in ((.5, "p50"), (.99, "p99")):
                est, lo, hi = A.bootstrap_diff_ci(r_ok.ttft_ms, v_ok.ttft_ms, q=q, n=2000)
                metrics.append(_metric(f"ttft_{tag}", f"Time to first token ({tag})", "ms", "lower", "A", ref, rival,
                                       float(r_ok.ttft_ms.quantile(q)), float(v_ok.ttft_ms.quantile(q)), ci=[lo, hi]))
            base = ta.baseline.dropna().iloc[0] if ta.baseline.notna().any() else None
            b_ok = ok[ok.system == base] if base else pd.DataFrame()
            if len(b_ok):
                bp50 = float(b_ok.ttft_ms.median())
                metrics.append(_metric("overhead_p50", "Added latency vs direct call (p50)", "ms", "lower", "A",
                                       ref, rival, float(r_ok.ttft_ms.median()) - bp50, float(v_ok.ttft_ms.median()) - bp50,
                                       note=f"compared with {base}"))
            tps_r, tps_v = r_ok.output_tps.dropna(), v_ok.output_tps.dropna()
            if len(tps_r) and len(tps_v):
                est, lo, hi = A.bootstrap_diff_ci(tps_r, tps_v, q=.5, n=2000)
                metrics.append(_metric("tps", "Output speed (median)", "tok/s", "higher", "A", ref, rival,
                                       float(tps_r.median()), float(tps_v.median()), ci=[lo, hi]))
            cpm_r, cpm_v = A._cost_per_mtok(r_ok), A._cost_per_mtok(v_ok)
            metrics.append(_metric("cost_mtok", "Cost per 1M tokens, same model", "$", "lower", "A", ref, rival,
                                   cpm_r, cpm_v, note="billed cost when reported, else list price"))
        r_all, v_all = ta[ta.system == ref], ta[ta.system == rival]
        if len(r_all) and len(v_all):
            metrics.append(_metric("success_a", "Success rate", "%", "higher", "A", ref, rival,
                                   float(r_all.ok.mean() * 100), float(v_all.ok.mean() * 100)))
        out["cdf"] = {}
        for (shape, stream), g in ok.groupby(["meta.shape", "stream"]):
            key = f"{shape}|{'stream' if stream else 'plain'}"
            out["cdf"][key] = {s: _cdf(sg.ttft_ms) for s, sg in g.groupby("system")}

    # ---------------- Track B
    tb = df[df.track == "B"].copy()
    if not tb.empty:
        b_tab = A.track_b_summary(df, reference=ref)
        out["track_b"] = _records(b_tab)
        tb["label"] = tb["meta.label"]
        r_b, v_b = tb[tb.label == ref], tb[tb.label == rival]
        if len(r_b) and len(v_b):
            rs, vs = r_b[r_b.correct.notna()], v_b[v_b.correct.notna()]
            p = A.mcnemar_exact(rs.set_index(["meta.workload", "meta.item_id"]).correct,
                                vs.set_index(["meta.workload", "meta.item_id"]).correct) if len(rs) and len(vs) else None
            acc_r = float(rs.correct.astype(float).mean() * 100) if len(rs) else None
            acc_v = float(vs.correct.astype(float).mean() * 100) if len(vs) else None
            metrics.append(_metric("accuracy", "Answer accuracy (routed)", "%", "higher", "B", ref, rival,
                                   acc_r, acc_v, p=p, note="paired McNemar test on the same questions"))

            def cpc(frame):
                c = frame.cost_usd.sum(min_count=1)
                n = int((frame.correct == True).sum())  # noqa: E712
                return float(c / n * 1000) if n and c == c else None

            def cpk(frame):
                c = frame.cost_usd.sum(min_count=1)
                return float(c / len(frame) * 1000) if len(frame) and c == c else None
            metrics.append(_metric("cost_correct", "Cost per 1,000 correct answers", "$", "lower", "B", ref, rival,
                                   cpc(r_b), cpc(v_b)))
            metrics.append(_metric("cost_1k", "Cost per 1,000 requests", "$", "lower", "B", ref, rival,
                                   cpk(r_b), cpk(v_b)))
        if not b_tab.empty:
            out["frontier"] = _records(b_tab[["workload", "label", "role", "accuracy", "acc_lo", "acc_hi",
                                              "cost_per_1k_usd", "pareto", "n"]])
            mix = {}
            for lab in (ref, rival):
                s = tb[tb.label == lab].model_served.fillna("unknown").value_counts()
                mix[lab] = [[k, int(v)] for k, v in s.items()]
            out["model_mix"] = mix

    # ---------------- Load
    if (df.track == "load").any():
        ld = A.load_summary(df, load_cfg.get("slo_p99_ttft_ms", 2000), load_cfg.get("max_error_rate", 0.01))
        out["load"] = _records(ld)
        out["load_slo_ms"] = load_cfg.get("slo_p99_ttft_ms", 2000)
        ms = A.max_sustained_rps(ld)
        if ref in set(ld.system) and rival in set(ld.system):
            metrics.append(_metric("max_rps", "Max sustained load (p99 within SLO)", "req/s", "higher", "load",
                                   ref, rival, float(ms.get(ref, 0.0)), float(ms.get(rival, 0.0))))

    decided = [m for m in metrics if m.get("winner") in (ref, rival)]
    out["score"] = {
        ref: sum(1 for m in decided if m["winner"] == ref),
        rival: sum(1 for m in decided if m["winner"] == rival),
        "significant_" + ref: sum(1 for m in decided if m["winner"] == ref and m.get("significant")),
        "significant_" + rival: sum(1 for m in decided if m["winner"] == rival and m.get("significant")),
        "total": len(metrics),
    }
    return clean(out)
