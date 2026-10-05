"""Command-line entry point: `gbench <command>`."""
from __future__ import annotations

import argparse
import asyncio
import sys

from .config import load_systems, select_systems


def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def _csv(s: str | None):
    return [x.strip() for x in s.split(",") if x.strip()] if s else None


def main(argv: list[str] | None = None) -> int:
    _load_env()
    p = argparse.ArgumentParser(prog="gbench", description="Benchmark LLM gateways and routers.")
    p.add_argument("--systems-config", default="configs/systems.yaml")
    p.add_argument("--out", default="results", help="directory for raw JSONL results")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="show configured systems and whether their API key is set")

    pa = sub.add_parser("track-a", help="gateway overhead (model pinned)")
    pa.add_argument("--config", default="configs/track_a.yaml")
    pa.add_argument("--systems", help="comma-separated subset")
    pa.add_argument("--requests", type=int, help="override requests_per_cell (e.g. 20 for a pilot)")
    pa.add_argument("--shapes", help="comma-separated subset of shapes")

    pb = sub.add_parser("track-b", help="routing quality per dollar")
    pb.add_argument("--config", default="configs/track_b.yaml")
    pb.add_argument("--systems", help="comma-separated subset of routers/baselines")
    pb.add_argument("--workloads", help="comma-separated subset of workloads")
    pb.add_argument("--max-items", type=int)

    pl = sub.add_parser("load", help="open-loop load test")
    pl.add_argument("--config", default="configs/load.yaml")
    pl.add_argument("--systems", help="comma-separated subset")
    pl.add_argument("--rates", help="comma-separated req/s steps (override)")
    pl.add_argument("--duration", type=float, help="seconds per step (override)")

    pd_ = sub.add_parser("prepare-data", help="download + convert public datasets (needs '.[data]')")
    pd_.add_argument("--dest", default="data/prepared")
    pd_.add_argument("--limit", type=int)

    pr = sub.add_parser("report", help="analyse results and write tables, charts and REPORT.md")
    pr.add_argument("inputs", nargs="*", default=["results"])
    pr.add_argument("--report-dir", default="reports/latest")
    pr.add_argument("--reference", default="concentrate", help="system that significance tests compare against")

    pu = sub.add_parser("ui", help="local web UI: Concentrate vs OpenRouter (needs '.[ui]')")
    pu.add_argument("--host", default="127.0.0.1")
    pu.add_argument("--port", type=int, default=8765)

    args = p.parse_args(argv)

    if args.cmd == "ui":
        from .ui.server import serve
        serve(args.host, args.port)
        return 0

    if args.cmd == "prepare-data":
        from .datasets import prepare_all
        print(prepare_all(args.dest, args.limit))
        return 0
    if args.cmd == "report":
        from .report import build_report
        build_report(args.inputs, args.report_dir, reference=args.reference)
        return 0

    systems = load_systems(args.systems_config)

    if args.cmd == "list":
        for s in systems.values():
            flag = "ok " if s.available else "-- "
            print(f"{flag}{s.name:24s} {s.kind:9s} tracks={','.join(s.tracks):4s} {s.base_url}")
        return 0
    if args.cmd == "track-a":
        from .track_a import run_track_a
        sel = select_systems(systems, "A", _csv(args.systems))
        asyncio.run(run_track_a(sel, args.config, args.out, args.requests, _csv(args.shapes)))
        return 0
    if args.cmd == "track-b":
        from .track_b import run_track_b
        asyncio.run(run_track_b(systems, args.config, args.out, _csv(args.systems), _csv(args.workloads), args.max_items))
        return 0
    if args.cmd == "load":
        from .load import run_load
        sel = select_systems(systems, None, _csv(args.systems))
        rates = [float(x) for x in _csv(args.rates)] if args.rates else None
        asyncio.run(run_load(sel, args.config, args.out, rates, args.duration))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
