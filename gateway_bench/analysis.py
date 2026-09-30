"""Statistics: bootstrap CIs, significance tests, Pareto frontier, AIQ, composite scorecard."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from .pricing import PriceBook

# ---------------------------------------------------------------- basic stats


def bootstrap_ci(x, stat: Callable = np.median, n: int = 10_000, alpha: float = 0.05, seed: int = 0):
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    boots = stat(x[idx], axis=1)
    return (float(stat(x)), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)))


def bootstrap_diff_ci(a, b, q: float = 0.5, n: int = 5_000, alpha: float = 0.05, seed: int = 0):
    """CI of quantile(a) - quantile(b) for independent samples."""
    a = np.asarray(pd.Series(a).dropna(), dtype=float)
    b = np.asarray(pd.Series(b).dropna(), dtype=float)
    if len(a) == 0 or len(b) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    ba = np.quantile(a[rng.integers(0, len(a), (n, len(a)))], q, axis=1)
    bb = np.quantile(b[rng.integers(0, len(b), (n, len(b)))], q, axis=1)
    d = ba - bb
    return (float(np.quantile(a, q) - np.quantile(b, q)), float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2)))


def holm(pvals: pd.Series) -> pd.Series:
    """Holm-Bonferroni adjusted p-values (NaNs are left as NaN)."""
    p = pvals.dropna().sort_values()
    m = len(p)
    adj = pd.Series(index=p.index, dtype=float)
    running = 0.0
    for i, (k, v) in enumerate(p.items()):
        running = max(running, min(1.0, (m - i) * v))
        adj[k] = running
    return adj.reindex(pvals.index)


def mcnemar_exact(a: pd.Series, b: pd.Series) -> float:
    """Paired exact McNemar test on boolean series aligned by index."""
    j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    n01 = int(((j.a == True) & (j.b == False)).sum())  # noqa: E712
    n10 = int(((j.a == False) & (j.b == True)).sum())  # noqa: E712
    if n01 + n10 == 0:
        return 1.0
    return float(stats.binomtest(n01, n01 + n10, 0.5).pvalue)


def add_costs(df: pd.DataFrame, prices: PriceBook) -> pd.DataFrame:
    if df.empty:
        return df
    if "cost_usd" not in df.columns or df["cost_usd"].isna().all():
        costs = [prices.cost(r) for r in df.to_dict("records")]
        df = df.copy()
        df["cost_usd"] = [c for c, _ in costs]
        df["cost_source"] = [s for _, s in costs]
    return df


# ---------------------------------------------------------------- Track A


def track_a_summary(df: pd.DataFrame, prices: PriceBook, baseline: str | None = None) -> pd.DataFrame:
    df = df[df["track"] == "A"].copy()
    if df.empty:
        return pd.DataFrame()
    df = df[df["meta.warmup"] != True]  # noqa: E712
    df = add_costs(df, prices)
    df["tokens"] = df["input_tokens"].fillna(0) + df["output_tokens"].fillna(0)
    baseline = baseline or (df["baseline"].dropna().iloc[0] if "baseline" in df and df["baseline"].notna().any() else None)

    rows = []
    for (shape, stream), g in df.groupby(["meta.shape", "stream"]):
        base_ok = g[(g.system == baseline) & g.ok]
        base_cpt = _cost_per_mtok(base_ok)
        for sys_name, s in g.groupby("system"):
            ok = s[s.ok]
            row = {
                "shape": shape, "stream": bool(stream), "system": sys_name, "n": len(s),
                "success_rate": float(s.ok.mean()),
                "ttft_p50_ms": _q(ok.ttft_ms, .5), "ttft_p90_ms": _q(ok.ttft_ms, .9), "ttft_p99_ms": _q(ok.ttft_ms, .99),
                "e2e_p50_ms": _q(ok.e2e_ms, .5), "e2e_p99_ms": _q(ok.e2e_ms, .99),
                "output_tps_p50": _q(ok.output_tps, .5),
                "cost_per_mtok_usd": _cost_per_mtok(ok),
                "served_model_mismatch": _mismatch(ok),
            }
            if baseline and sys_name != baseline and len(base_ok):
                for q, tag in ((.5, "p50"), (.99, "p99")):
                    est, lo, hi = bootstrap_diff_ci(ok.ttft_ms, base_ok.ttft_ms, q=q)
                    row[f"overhead_{tag}_ms"], row[f"overhead_{tag}_lo"], row[f"overhead_{tag}_hi"] = est, lo, hi
                if ok.ttft_ms.notna().sum() and base_ok.ttft_ms.notna().sum():
                    row["mw_p"] = float(stats.mannwhitneyu(ok.ttft_ms.dropna(), base_ok.ttft_ms.dropna(),
                                                           alternative="two-sided").pvalue)
                cpt = row["cost_per_mtok_usd"]
                row["effective_markup"] = (cpt / base_cpt - 1) if (base_cpt and cpt == cpt and cpt is not None) else np.nan
            rows.append(row)
    out = pd.DataFrame(rows)
    if "mw_p" in out:
        out["mw_p_holm"] = holm(out["mw_p"])
    return out.sort_values(["shape", "stream", "ttft_p50_ms"]).reset_index(drop=True)


def _q(s, q):
    s = pd.Series(s).dropna()
    return float(s.quantile(q)) if len(s) else np.nan


def _cost_per_mtok(g: pd.DataFrame):
    if g.empty or g["cost_usd"].isna().all():
        return np.nan
    tok = (g["input_tokens"].fillna(0) + g["output_tokens"].fillna(0)).sum()
    return float(g["cost_usd"].sum() / tok * 1e6) if tok else np.nan


def _mismatch(g: pd.DataFrame) -> float:
    if g.empty or "model_served" not in g:
        return np.nan
    req = g["model_requested"].astype(str).str.split("/").str[-1].str.lower()
    srv = g["model_served"].fillna("").astype(str).str.lower()
    return float(np.mean([not (r in s) for r, s in zip(req, srv)]))


# ---------------------------------------------------------------- Track B


def track_b_summary(df: pd.DataFrame, reference: str = "concentrate") -> pd.DataFrame:
    df = df[df["track"] == "B"].copy()
    if df.empty:
        return pd.DataFrame()
    df["label"] = df["meta.label"]
    rows = []
    for wl, g in df.groupby("meta.workload"):
        per = {}
        for label, s in g.groupby("label"):
            scored = s[s.correct.notna()]
            acc, lo, hi = bootstrap_ci(scored.correct.astype(float), stat=np.mean) if len(scored) else (np.nan,) * 3
            total_cost = s.cost_usd.sum(min_count=1)
            n_correct = int(scored.correct.astype(bool).sum())
            per[label] = {
                "workload": wl, "label": label, "role": s["meta.role"].iloc[0], "n": len(s),
                "success_rate": float(s.ok.mean()), "accuracy": acc, "acc_lo": lo, "acc_hi": hi,
                "cost_total_usd": total_cost,
                "cost_per_1k_usd": (total_cost / len(s) * 1000) if total_cost == total_cost else np.nan,
                "cost_per_correct_usd": (total_cost / n_correct) if n_correct and total_cost == total_cost else np.nan,
                "ttft_p50_ms": _q(s[s.ok].ttft_ms, .5),
                "models_used": ", ".join(f"{m}:{c}" for m, c in s.model_served.fillna("?").value_counts().head(5).items()),
            }
        base = [v for v in per.values() if v["role"] == "baseline" and v["accuracy"] == v["accuracy"]]
        best = max(base, key=lambda v: v["accuracy"]) if base else None
        ref_series = g[g.label == reference].set_index("meta.item_id").correct
        for label, v in per.items():
            if best:
                v["quality_retained"] = v["accuracy"] / best["accuracy"] if best["accuracy"] else np.nan
                v["cost_saved_vs_strongest"] = 1 - v["cost_total_usd"] / best["cost_total_usd"] if best["cost_total_usd"] else np.nan
            if label != reference and len(ref_series):
                other = g[g.label == label].set_index("meta.item_id").correct
                v[f"mcnemar_p_vs_{reference}"] = mcnemar_exact(ref_series, other)
            rows.append(v)
    out = pd.DataFrame(rows)
    col = f"mcnemar_p_vs_{reference}"
    if col in out:
        out[col + "_holm"] = holm(out[col])
    out["pareto"] = False
    for wl, g in out.groupby("workload"):
        out.loc[g.index, "pareto"] = pareto_mask(g.cost_per_1k_usd.values, g.accuracy.values)
    return out.sort_values(["workload", "accuracy"], ascending=[True, False]).reset_index(drop=True)


def pareto_mask(cost, quality) -> np.ndarray:
    """True where no other point is cheaper-or-equal AND better-or-equal (strictly better in one)."""
    cost = np.asarray(cost, float)
    quality = np.asarray(quality, float)
    mask = np.ones(len(cost), bool)
    for i in range(len(cost)):
        if np.isnan(cost[i]) or np.isnan(quality[i]):
            mask[i] = False
            continue
        dom = (cost <= cost[i]) & (quality >= quality[i]) & ((cost < cost[i]) | (quality > quality[i]))
        mask[i] = not dom.any()
    return mask


def aiq(points: list[tuple[float, float]], c_min: float, c_max: float) -> float:
    """RouterBench-style AIQ: area under the non-decreasing convex hull of (cost, quality)
    points over [c_min, c_max], normalised by the range. Points = one router at several
    cost/quality settings (e.g. thresholds)."""
    pts = sorted((c, q) for c, q in points if c == c and q == q)
    if not pts or c_max <= c_min:
        return float("nan")
    hull: list[tuple[float, float]] = []
    for p in pts:  # upper hull
        while len(hull) >= 2 and _cross(hull[-2], hull[-1], p) >= 0:
            hull.pop()
        hull.append(p)
    xs, ys, best = [], [], -np.inf
    for c, q in hull:
        best = max(best, q)
        xs.append(c)
        ys.append(best)
    grid = np.linspace(c_min, c_max, 512)
    vals = np.interp(grid, xs, ys, left=np.nan, right=ys[-1])
    vals = np.where(grid < xs[0], 0.0, vals)   # no quality before the cheapest point exists
    _trap = getattr(np, "trapezoid", None) or np.trapz
    return float(_trap(vals, grid) / (c_max - c_min))


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def aiq_table(b: pd.DataFrame) -> pd.DataFrame:
    """Group labels like 'concentrate@cheap', 'concentrate@balanced' into one router curve."""
    if b.empty:
        return pd.DataFrame()
    rows = []
    for wl, g in b.groupby("workload"):
        base = g[g.role == "baseline"]
        if base.empty:
            continue
        c_min, c_max = base.cost_per_1k_usd.min(), base.cost_per_1k_usd.max()
        fam = g[g.role == "router"].assign(family=lambda x: x.label.str.split("@").str[0])
        for f, fg in fam.groupby("family"):
            rows.append({"workload": wl, "router": f, "points": len(fg),
                         "aiq": aiq(list(zip(fg.cost_per_1k_usd, fg.accuracy)), c_min, c_max)})
        rows.append({"workload": wl, "router": "baselines (hull)", "points": len(base),
                     "aiq": aiq(list(zip(base.cost_per_1k_usd, base.accuracy)), c_min, c_max)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Load


def load_summary(df: pd.DataFrame, slo_p99_ttft_ms: float = 2000, max_error_rate: float = 0.01) -> pd.DataFrame:
    df = df[df["track"] == "load"].copy()
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (s, rate), g in df.groupby(["system", "meta.target_rps"]):
        span = g.t_start.max() - g.t_start.min()
        ok = g[g.ok]
        rows.append({"system": s, "target_rps": rate, "achieved_rps": len(g) / span if span > 0 else np.nan,
                     "error_rate": 1 - g.ok.mean(), "ttft_p50_ms": _q(ok.ttft_ms, .5), "ttft_p99_ms": _q(ok.ttft_ms, .99)})
    out = pd.DataFrame(rows)
    out["meets_slo"] = (out.ttft_p99_ms <= slo_p99_ttft_ms) & (out.error_rate <= max_error_rate)
    return out.sort_values(["system", "target_rps"]).reset_index(drop=True)


def max_sustained_rps(load: pd.DataFrame) -> pd.Series:
    if load.empty:
        return pd.Series(dtype=float)
    return load[load.meets_slo].groupby("system").target_rps.max()


# ---------------------------------------------------------------- Scorecard


def _scale(s: pd.Series, higher_is_better: bool) -> pd.Series:
    s = s.astype(float)
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or hi == lo:
        return pd.Series(50.0, index=s.index)
    x = (s - lo) / (hi - lo) * 100
    return x if higher_is_better else 100 - x


def scorecard(a: pd.DataFrame, b: pd.DataFrame, weights: dict[str, float],
              features: dict[str, dict[str, bool]] | None = None) -> pd.DataFrame:
    """Per-system composite score. Systems missing a dimension are scored on the rest (weights renormalised)."""
    dims: dict[str, pd.Series] = {}
    if not a.empty and "overhead_p50_ms" in a:
        a = a[a.overhead_p50_ms.notna()]          # the direct-provider baseline is a reference, not a contender
    if not a.empty:
        ga = a.groupby("system")
        speed = pd.concat([
            _scale(ga.overhead_p50_ms.mean(), False) if "overhead_p50_ms" in a else None,
            _scale(ga.overhead_p99_ms.mean(), False) if "overhead_p99_ms" in a else None,
            _scale(ga.output_tps_p50.mean(), True),
        ], axis=1).mean(axis=1)
        dims["speed"] = speed
        if "effective_markup" in a and a.effective_markup.notna().any():
            dims["cost"] = _scale(ga.effective_markup.mean(), False)
        dims["reliability"] = _scale(ga.success_rate.mean(), True)
    if not b.empty:
        rb = b[b.role == "router"].groupby("label")
        eff = pd.concat([_scale(rb.cost_per_correct_usd.mean(), False),
                         _scale(rb.quality_retained.mean(), True) if "quality_retained" in b else None], axis=1).mean(axis=1)
        dims["efficiency"] = eff
    if features:
        dims["features"] = pd.Series({k: 100 * np.mean(list(v.values())) for k, v in features.items() if v})
    table = pd.DataFrame(dims)
    w = pd.Series({k: v for k, v in weights.items() if k in table.columns})
    present = table.notna().mul(w, axis=1)
    table["score"] = table[w.index].fillna(0).mul(w).sum(axis=1) / present.sum(axis=1)
    table["dims_covered"] = table[w.index].notna().sum(axis=1)
    return table.sort_values("score", ascending=False)
