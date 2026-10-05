"""Build the JSON the UI renders: head-to-head metrics, chart series and tables for one run."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import analysis as A
from ..pricing import PriceBook
from ..results import read_results

_META_COLS = ("meta.rep", "meta.warmup", "meta.shape", "meta.workload", "meta.label", "meta.role", "meta.item_id",
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
    if a_val == b_val or (max(abs(a_val), abs(b_val)) and abs(a_val - b_val) / max(abs(a_val), abs(b_val)) < 0.005):
        m["winner"] = "tie"            # gaps under 0.5% are noise, not a win
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


def readable_error(err) -> str:
    """'status 400: {json}' -> 'status 400: <innermost message>' (OpenRouter nests the upstream
    provider's error as a JSON string in error.metadata.raw)."""
    if not isinstance(err, str) or not err:
        return "unknown error"
    head, _, rest = err.partition(": ")
    i = rest.find("{")
    if not head.startswith("status") or i < 0:
        return err[:400]
    try:
        body = json.loads(rest[i:])
    except ValueError:
        # truncated JSON: pull "message" values out directly (outer first, innermost last)
        msgs = [m.replace('\\"', '"') for m in re.findall(r'\\?"message\\?"\s*:\s*\\?"((?:[^"\\]|\\.)*?)\\?"', rest)]
        if not msgs:
            return err[:400]
        return f"{head}: {msgs[0]}" + (f" (upstream: {msgs[-1]})" if len(msgs) > 1 else "")
    e = body.get("error", body) if isinstance(body, dict) else {}
    msg = e.get("message") if isinstance(e, dict) else None
    raw = (e.get("metadata") or {}).get("raw") if isinstance(e, dict) else None
    if isinstance(raw, str):
        try:
            inner = json.loads(raw)
            inner = inner.get("error", inner)
            if isinstance(inner, dict) and inner.get("message"):
                msg = f"{msg} (upstream: {inner['message']})" if msg else inner["message"]
        except ValueError:
            pass
    return f"{head}: {msg}" if msg else err[:400]


def _catalog_price(catalog: dict, model) -> dict | None:
    if not catalog or not isinstance(model, str) or not model:
        return None
    if model in catalog:
        return catalog[model]
    tail = model.split("/", 1)[-1]
    hits = [v for k, v in catalog.items() if k.split("/", 1)[-1] == tail]
    return hits[0] if len(hits) == 1 else None


def fill_costs(df: pd.DataFrame, prices: PriceBook, catalog: dict | None) -> pd.DataFrame:
    """cost_usd per row: billed cost if reported, else configured list price, else the
    OpenRouter catalog list price for the served model (marked as an estimate)."""
    costs, srcs = [], []
    for r in df.to_dict("records"):
        rep = r.get("cost_reported")
        if rep is not None and rep == rep:
            costs.append(float(rep)); srcs.append("reported"); continue
        c, src = prices.cost({**r, "cost_reported": None})
        if c is None:
            pr = (_catalog_price(catalog or {}, r.get("model_served"))
                  or _catalog_price(catalog or {}, r.get("model_requested")))
            it, ot = r.get("input_tokens"), r.get("output_tokens")
            if pr and it == it and ot == ot and it is not None and ot is not None:
                c, src = (it * pr["input"] + ot * pr["output"]) / 1e6, "catalog_estimate"
        costs.append(c); srcs.append(src)
    df = df.copy()
    df["cost_usd"], df["cost_source"] = costs, srcs
    return df


def _health(df: pd.DataFrame) -> list[dict]:
    rows = []
    d = df[df["meta.warmup"] != True]  # noqa: E712
    for track, g in d.groupby("track"):
        key = "meta.label" if track == "B" else "system"
        for name, s in g.groupby(key):
            failed = s[s.ok != True]  # noqa: E712
            top = failed.error.map(readable_error).value_counts().head(3)
            rows.append({"track": track, "system": name, "n": int(len(s)), "failed": int(len(failed)),
                         "by_status": {str(k): int(v) for k, v in failed.status.fillna("none").astype(str).value_counts().items()},
                         "top_errors": [{"message": m, "count": int(c)} for m, c in top.items()]})
    return rows


def nice_pct(x) -> str:
    return f"{x * 100:.0f}%"


def _boot_median_ci(x, n: int = 2000, seed: int = 0):
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    if len(x) < 2:
        return [None, None]
    rng = np.random.default_rng(seed)
    b = np.median(x[rng.integers(0, len(x), (n, len(x)))], axis=1)
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def paired_requests(ta: pd.DataFrame, ref: str, rival: str) -> pd.DataFrame:
    """Track A requests both gateways completed in the same cell and round (shape, stream, rep).
    Comparing only these removes bias when one side failed some requests (e.g. out of credits)."""
    key = ["meta.shape", "stream", "meta.rep"]
    ok = ta[(ta.ok == True) & ta["meta.rep"].notna()]  # noqa: E712
    cols = ["ttft_ms", "e2e_ms", "input_tokens", "output_tokens", "cost_usd"]
    r = ok[ok.system == ref].drop_duplicates(key).set_index(key)[cols]
    v = ok[ok.system == rival].drop_duplicates(key).set_index(key)[cols]
    return r.join(v, lsuffix="_r", rsuffix="_v", how="inner").reset_index()


def _row_list_cost(r: dict, catalog: dict | None):
    pr = _catalog_price(catalog or {}, r.get("model_served")) or _catalog_price(catalog or {}, r.get("model_requested"))
    it, ot = r.get("input_tokens"), r.get("output_tokens")
    if pr and it is not None and ot is not None and it == it and ot == ot:
        return (it * pr["input"] + ot * pr["output"]) / 1e6
    rep = r.get("cost_reported")
    return float(rep) if rep is not None and rep == rep else None


def billing_analysis(df: pd.DataFrame, catalog: dict | None, billing: dict, ref: str, rival: str,
                     pairs: pd.DataFrame | None) -> dict:
    """Compare what each gateway actually charged (entered from its dashboard, or reported per request)
    with list price for the tokens it processed, then price the same work for both."""
    ok = df[df.ok == True]  # noqa: E712
    sides = {}
    for s_ in (ref, rival):
        g = ok[ok.system == s_]
        costs = [_row_list_cost(r, catalog) for r in g.to_dict("records")]
        known = [c for c in costs if c is not None]
        reported = g.cost_reported.dropna()
        billed, src = billing.get(s_), "entered"
        if billed is None and len(reported) and len(reported) == len(g):
            billed, src = float(reported.sum()), "reported by API"
        if billed is None:
            src = None
        lst = float(sum(known)) if known else None
        sides[s_] = {"requests": int(len(g)), "requests_failed": int((df.system == s_).sum() - len(g)),
                     "tokens_in": int(g.input_tokens.fillna(0).sum()), "tokens_out": int(g.output_tokens.fillna(0).sum()),
                     "list_cost": lst, "unpriced": int(len(costs) - len(known)), "billed": billed, "billed_source": src,
                     "reported_total": float(reported.sum()) if len(reported) else None,
                     "ratio": (billed / lst) if (billed is not None and lst) else None}
    same = None
    if pairs is not None and len(pairs):
        # list cost of the work both completed: matched Track A requests, priced per side then averaged
        ta_ok = ok[ok.track == "A"]
        cells = []
        for (shape, stream), g in ta_ok.groupby(["meta.shape", "stream"]):
            n_r, n_v = int((g.system == ref).sum()), int((g.system == rival).sum())
            per = [c for c in (_row_list_cost(x, catalog) for x in g.to_dict("records")) if c is not None]
            if per and min(n_r, n_v):
                cells.append(min(n_r, n_v) * float(np.mean(per)))
        same_list = float(sum(cells)) if cells else None
        same = {"requests": int(len(pairs)), "list_cost": same_list,
                "cost": {k: (v["ratio"] * same_list if (v["ratio"] is not None and same_list) else None) for k, v in sides.items()}}
    return {"sides": sides, "same_work": same, "note": billing.get("note") or ""}


def build_summary(run_dir: str | Path, prices_path: str | Path, ref: str = "concentrate",
                  rival: str = "openrouter", load_cfg: dict | None = None, catalog: dict | None = None,
                  billing: dict | None = None) -> dict:
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
    df = fill_costs(df, prices, catalog)
    load_cfg = load_cfg or {}
    out: dict[str, Any] = {"empty": False, "ref": ref, "rival": rival, "metrics": []}
    metrics = out["metrics"]
    health = _health(df)
    out["health"] = health
    out["errors"] = [h for h in health if h["failed"]]

    def cost_note(frames: dict) -> str:
        est = [nm for nm, f in frames.items() if f.cost_source.isin(["catalog_estimate", "list_price"]).any()]
        unk = [nm for nm, f in frames.items() if len(f) and f.cost_usd.isna().all()]
        parts = []
        if est:
            parts.append(f"{', '.join(est)}: estimated from list price (no billed cost returned)")
        if unk:
            parts.append(f"{', '.join(unk)}: cost unknown")
        return "; ".join(parts) or "billed cost reported by each gateway"

    # ---------------- Track A
    ta = df[(df.track == "A") & (df["meta.warmup"] != True)].copy()  # noqa: E712
    if not ta.empty:
        a_tab = A.track_a_summary(df, prices)
        out["track_a"] = _records(a_tab)
        ta = A.add_costs(ta, prices)
        ok_all = ta[ta.ok == True]  # noqa: E712
        # If one side failed noticeably more (e.g. ran out of credits), compare speed only on the requests
        # both completed: the failures cluster on some request types and would skew every percentile.
        fail = {k: 1 - ta[ta.system == k].ok.mean() for k in (ref, rival) if (ta.system == k).any()}
        basis = "all"
        ok = ok_all
        if len(fail) == 2 and abs(fail[ref] - fail[rival]) > 0.02:
            keys = paired_requests(ta, ref, rival)[["meta.shape", "stream", "meta.rep"]]
            if len(keys):
                ok = ok_all.merge(keys, on=["meta.shape", "stream", "meta.rep"], how="inner")
                basis = "matched"
        out["latency_basis"] = basis
        basis_note = f"matched requests only ({int((ok.system == ref).sum())} each), since failure rates differ" if basis == "matched" else None
        r_ok, v_ok = ok[ok.system == ref], ok[ok.system == rival]
        # time to first token only means something when streaming (non-streaming "first token" = whole answer)
        streamed = ok[ok.stream == True]  # noqa: E712
        tt = streamed if len(streamed) else ok
        r_tt, v_tt = tt[tt.system == ref], tt[tt.system == rival]
        tt_note = (f"streaming requests ({int((tt.system == ref).sum())} each)" if len(streamed)
                   else "non-streaming: first token = full response") + \
            ("; matched requests only, since failure rates differ" if basis == "matched" else "")
        if len(r_ok) and len(v_ok):
            for q, tag in ((.5, "p50"), (.9, "p90"), (.99, "p99")):
                est, lo, hi = A.bootstrap_diff_ci(r_tt.ttft_ms, v_tt.ttft_ms, q=q, n=2000)
                metrics.append(_metric(f"ttft_{tag}", f"Time to first token ({tag})", "ms", "lower", "A", ref, rival,
                                       float(r_tt.ttft_ms.quantile(q)), float(v_tt.ttft_ms.quantile(q)), ci=[lo, hi],
                                       note=tt_note))
            base = ta.baseline.dropna().iloc[0] if ta.baseline.notna().any() else None
            b_ok = ok[ok.system == base] if base else pd.DataFrame()
            if len(b_ok):
                bp50 = float(b_ok.ttft_ms.median())
                metrics.append(_metric("overhead_p50", "Added latency vs direct call (p50)", "ms", "lower", "A",
                                       ref, rival, float(r_ok.ttft_ms.median()) - bp50, float(v_ok.ttft_ms.median()) - bp50,
                                       note=f"compared with {base}"))
            for q, tag in ((.5, "p50"), (.99, "p99")):
                est, lo, hi = A.bootstrap_diff_ci(r_ok.e2e_ms, v_ok.e2e_ms, q=q, n=2000)
                metrics.append(_metric(f"e2e_{tag}", f"Total response time ({tag})", "ms", "lower", "A", ref, rival,
                                       float(r_ok.e2e_ms.quantile(q)), float(v_ok.e2e_ms.quantile(q)), ci=[lo, hi],
                                       note="request sent to last byte" + (f"; {basis_note}" if basis_note else "")))
            def _spread(g):
                t = g.ttft_ms.dropna()
                return float(t.quantile(.99) / t.median()) if len(t) and t.median() else None
            metrics.append(_metric("consistency", "Tail consistency (TTFT p99 ÷ p50)", "×", "lower", "A", ref, rival,
                                   _spread(r_tt), _spread(v_tt), note="1× = every request as fast as the typical one; " + tt_note))
            # slow outliers: TTFT above 2x the combined median of the same cell (size + mode)
            med = tt.groupby(["meta.shape", "stream"]).ttft_ms.median().rename("cell_med")
            okm = tt.join(med, on=["meta.shape", "stream"])
            def _slow(g):
                g = g[g.ttft_ms.notna()]
                return float((g.ttft_ms > 2 * g.cell_med).mean() * 100) if len(g) else None
            metrics.append(_metric("slow_share", "Slow outliers (TTFT over 2× typical)", "%", "lower", "A", ref, rival,
                                   _slow(okm[okm.system == ref]), _slow(okm[okm.system == rival]),
                                   note="typical = median of both gateways for the same size and mode"))
            tps_r, tps_v = r_ok.output_tps.dropna(), v_ok.output_tps.dropna()
            if len(tps_r) and len(tps_v):
                est, lo, hi = A.bootstrap_diff_ci(tps_r, tps_v, q=.5, n=2000)
                metrics.append(_metric("tps", "Output speed (median)", "tok/s", "higher", "A", ref, rival,
                                       float(tps_r.median()), float(tps_v.median()), ci=[lo, hi]))
            cpm_r, cpm_v = A._cost_per_mtok(r_ok), A._cost_per_mtok(v_ok)
            metrics.append(_metric("cost_mtok", "Cost per 1M tokens, same model", "$", "lower", "A", ref, rival,
                                   cpm_r, cpm_v, note=cost_note({ref: r_ok, rival: v_ok})))
        r_all, v_all = ta[ta.system == ref], ta[ta.system == rival]
        if len(r_all) and len(v_all):
            sr, sv = float(r_all.ok.mean() * 100), float(v_all.ok.mean() * 100)
            m = _metric("success_a", "Success rate", "%", "higher", "A", ref, rival, sr, sv)
            if sr == 0 and sv == 0:
                m.update(winner=None, note="no request succeeded on either side, see Errors")
            metrics.append(m)
        pairs = paired_requests(ta, ref, rival)
        out["pairs_n"] = int(len(pairs))
        if len(pairs):
            ps = pairs[pairs.stream == True] if (pairs.stream == True).any() else pairs  # noqa: E712
            d = (ps.ttft_ms_r - ps.ttft_ms_v).dropna()
            if len(d):
                metrics.append(_metric("ttft_matched", "Head-to-head: time to first token on the same requests", "ms", "lower",
                                       "A", ref, rival, float(ps.ttft_ms_r.median()), float(ps.ttft_ms_v.median()),
                                       ci=_boot_median_ci(d),
                                       note=f"{len(d)} streaming request pairs; {ref.title()} got the first token sooner in {nice_pct((d < 0).mean())} of them"))
            tr = (pairs.input_tokens_r.fillna(0) + pairs.output_tokens_r.fillna(0)).mean()
            tv = (pairs.input_tokens_v.fillna(0) + pairs.output_tokens_v.fillna(0)).mean()
            metrics.append(_metric("tokens_counted", "Tokens counted per matched request", "tok", "lower", "A", ref, rival,
                                   float(tr), float(tv), note="same prompts and limits; a gap means different token accounting (billing)"))
            rows = []
            for (shape, stream), g in pairs.groupby(["meta.shape", "stream"]):
                rows.append({"shape": shape, "stream": bool(stream), "pairs": int(len(g)),
                             "ttft_r": float(g.ttft_ms_r.median()) if g.ttft_ms_r.notna().any() else None,
                             "ttft_v": float(g.ttft_ms_v.median()) if g.ttft_ms_v.notna().any() else None,
                             "e2e_r": float(g.e2e_ms_r.median()), "e2e_v": float(g.e2e_ms_v.median()),
                             "faster_share": (lambda b: float((b.ttft_ms_r < b.ttft_ms_v).mean() * 100) if len(b) else None)(
                                 g[g.ttft_ms_r.notna() & g.ttft_ms_v.notna()])})
            out["paired"] = rows
        out["cdf"] = {}
        for (shape, stream), g in ok_all.groupby(["meta.shape", "stream"]):
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
            cn = cost_note({ref: r_b[r_b.ok == True], rival: v_b[v_b.ok == True]})  # noqa: E712
            metrics.append(_metric("cost_correct", "Cost per 1,000 correct answers", "$", "lower", "B", ref, rival,
                                   cpc(r_b), cpc(v_b), note=cn))
            metrics.append(_metric("cost_1k", "Cost per 1,000 requests", "$", "lower", "B", ref, rival,
                                   cpk(r_b), cpk(v_b), note=cn))
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

    ta_all = df[(df.track == "A") & (df["meta.warmup"] != True)]  # noqa: E712
    bill = billing_analysis(df, catalog, billing or {}, ref, rival,
                            paired_requests(ta_all, ref, rival) if len(ta_all) else None)
    out["billing"] = bill
    sd = bill["sides"]
    if sd[ref]["ratio"] is not None and sd[rival]["ratio"] is not None:
        metrics.append(_metric("billed_ratio", "Charged ÷ list price", "×", "lower", "billing", ref, rival,
                               sd[ref]["ratio"], sd[rival]["ratio"], note="1× = list price for the tokens processed"))
        sw = bill.get("same_work") or {}
        c = sw.get("cost") or {}
        if c.get(ref) is not None and c.get(rival) is not None:
            metrics.append(_metric("billed_same_work", "Charged for identical work", "$", "lower", "billing", ref, rival,
                                   c[ref], c[rival], note=f"{sw['requests']} requests both completed, at each one's actual charge rate"))

    decided = [m for m in metrics if m.get("winner") in (ref, rival)]
    out["score"] = {
        ref: sum(1 for m in decided if m["winner"] == ref),
        rival: sum(1 for m in decided if m["winner"] == rival),
        "significant_" + ref: sum(1 for m in decided if m["winner"] == ref and m.get("significant")),
        "significant_" + rival: sum(1 for m in decided if m["winner"] == rival and m.get("significant")),
        "total": len(metrics),
    }
    return clean(out)
