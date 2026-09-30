"""Turn raw results into CSV tables, charts and a Markdown summary."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import analysis as A  # noqa: E402
from .config import load_yaml  # noqa: E402
from .pricing import PriceBook  # noqa: E402
from .results import read_results  # noqa: E402

HIGHLIGHT = "concentrate"


def build_report(inputs: list[str], out_dir: str | Path = "reports/latest", configs_dir: str | Path = "configs",
                 reference: str = HIGHLIGHT, log=print) -> Path:
    out = Path(out_dir)
    (out / "charts").mkdir(parents=True, exist_ok=True)
    cfg = Path(configs_dir)
    prices = PriceBook.load(cfg / "prices.yaml")
    sc_cfg = load_yaml(cfg / "scorecard.yaml")
    load_cfg = load_yaml(cfg / "load.yaml") if (cfg / "load.yaml").exists() else {}

    df = read_results(inputs)
    if df.empty:
        raise SystemExit("no results found")
    for col in ("meta.warmup", "meta.shape", "meta.workload", "meta.label", "meta.role", "meta.item_id",
                "meta.target_rps", "baseline", "correct", "cost_usd"):
        if col not in df:
            df[col] = np.nan

    a = A.track_a_summary(df, prices) if (df.track == "A").any() else pd.DataFrame()
    b = A.track_b_summary(df, reference=reference) if (df.track == "B").any() else pd.DataFrame()
    q = A.aiq_table(b) if not b.empty else pd.DataFrame()
    ld = A.load_summary(df, load_cfg.get("slo_p99_ttft_ms", 2000), load_cfg.get("max_error_rate", 0.01)) \
        if (df.track == "load").any() else pd.DataFrame()
    sc = A.scorecard(a, b, sc_cfg.get("weights", {}), sc_cfg.get("feature_matrix") or None)
    alts = {name: A.scorecard(a, b, w, sc_cfg.get("feature_matrix") or None)["score"]
            for name, w in (sc_cfg.get("alternatives") or {}).items()}

    for name, t in (("track_a", a), ("track_b", b), ("aiq", q), ("load", ld), ("scorecard", sc)):
        if not t.empty:
            t.to_csv(out / f"{name}.csv", index=name == "scorecard")

    if not df[df.track == "A"].empty:
        _latency_cdfs(df[(df.track == "A") & (df["meta.warmup"] != True) & df.ok], out / "charts")  # noqa: E712
    if not b.empty:
        _frontiers(b, out / "charts", reference)
    if not ld.empty:
        _load_chart(ld, out / "charts")

    md = ["# Gateway benchmark report", "", f"Inputs: {', '.join(inputs)}", ""]
    if not sc.empty:
        md += ["## Composite scorecard", "", _md(sc.round(1).reset_index().rename(columns={"index": "system"})), ""]
        if alts:
            ranks = pd.DataFrame({"main": sc["score"].rank(ascending=False), **{k: v.rank(ascending=False) for k, v in alts.items()}})
            md += ["### Rank under alternative weights (sensitivity)", "", _md(ranks.reset_index().rename(columns={"index": "system"})), ""]
    if not a.empty:
        cols = ["shape", "stream", "system", "n", "success_rate", "ttft_p50_ms", "ttft_p99_ms",
                "overhead_p50_ms", "overhead_p50_lo", "overhead_p50_hi", "overhead_p99_ms", "effective_markup", "mw_p_holm"]
        md += ["## Track A: gateway overhead", "", _md(a[[c for c in cols if c in a]].round(3)), ""]
    if not b.empty:
        cols = ["workload", "label", "role", "n", "accuracy", "acc_lo", "acc_hi", "cost_per_1k_usd",
                "cost_per_correct_usd", "quality_retained", "cost_saved_vs_strongest", "pareto",
                f"mcnemar_p_vs_{reference}_holm", "models_used"]
        md += ["## Track B: routing quality per dollar", "", _md(b[[c for c in cols if c in b]].round(6)), ""]
    if not q.empty:
        md += ["### AIQ (area under cost-quality curve)", "", _md(q.round(4)), ""]
    if not ld.empty:
        md += ["## Load test", "", _md(ld.round(2)), "", "Max sustained RPS:", "",
               _md(A.max_sustained_rps(ld).reset_index()), ""]
    md += ["## Charts", ""] + [f"![{p.stem}](charts/{p.name})" for p in sorted((out / "charts").glob("*.png"))]
    (out / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    log(f"Report -> {out / 'REPORT.md'}")
    return out


def _md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_no data_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


def _color(name: str):
    return "#1f6feb" if name.startswith(HIGHLIGHT) else "#8b949e"


def _latency_cdfs(df: pd.DataFrame, out: Path):
    for (shape, stream), g in df.groupby(["meta.shape", "stream"]):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for s, sg in g.groupby("system"):
            x = np.sort(sg.ttft_ms.dropna().values)
            if len(x):
                ax.plot(x, np.arange(1, len(x) + 1) / len(x), label=s, color=_color(s),
                        lw=2.2 if s.startswith(HIGHLIGHT) else 1.1, alpha=1 if s.startswith(HIGHLIGHT) else .75)
        ax.set_xscale("log")
        ax.set_xlabel("Time to first token (ms, log scale)")
        ax.set_ylabel("Share of requests")
        ax.set_title(f"TTFT distribution: {shape}, {'streaming' if stream else 'non-streaming'}")
        ax.grid(alpha=.3)
        ax.legend(fontsize=7, ncol=2, frameon=False)
        fig.tight_layout()
        fig.savefig(out / f"ttft_cdf_{shape}_{'stream' if stream else 'plain'}.png", dpi=150)
        plt.close(fig)


def _frontiers(b: pd.DataFrame, out: Path, reference: str):
    for wl, g in b.groupby("workload"):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for _, r in g.iterrows():
            is_ref = str(r.label).startswith(reference)
            marker = "o" if r.role == "router" else "s"
            ax.scatter(r.cost_per_1k_usd, r.accuracy, s=70 if is_ref else 40, marker=marker,
                       color=_color(str(r.label)), edgecolor="black" if r.pareto else "none", zorder=3)
            ax.annotate(r.label, (r.cost_per_1k_usd, r.accuracy), fontsize=7, xytext=(4, 3), textcoords="offset points")
        f = g[g.pareto].sort_values("cost_per_1k_usd")
        ax.plot(f.cost_per_1k_usd, f.accuracy, ls="--", color="#555", lw=1, label="Pareto frontier")
        ax.set_xlabel("Cost per 1,000 requests (USD)")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Cost vs quality: {wl} (circles = routers, squares = single models)")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(out / f"frontier_{wl}.png", dpi=150)
        plt.close(fig)


def _load_chart(ld: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for s, g in ld.groupby("system"):
        ax.plot(g.target_rps, g.ttft_p99_ms, marker="o", label=s, color=_color(s))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Target arrival rate (req/s)")
    ax.set_ylabel("p99 TTFT (ms)")
    ax.set_title("Tail latency under load")
    ax.grid(alpha=.3)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "load_p99.png", dpi=150)
    plt.close(fig)
