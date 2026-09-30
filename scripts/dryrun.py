"""End-to-end dry run against local mock gateways (no API keys, no cost, ~1 minute).

    python scripts/dryrun.py

Starts 4 mock gateways, runs Track A, Track B and a short load test, then writes a report
to reports/dryrun/. Use it to check the pipeline before spending real money.
"""
from __future__ import annotations

import os
import shutil
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

from mock_gateway import serve  # noqa: E402

from gateway_bench.cli import main  # noqa: E402

MOCKS = [  # port, base latency, accuracy, $/1M tokens, name
    (9001, 40, 0.60, 0.5, "direct-small"),
    (9002, 60, 0.95, 10.0, "direct-frontier"),
    (9003, 44, 0.90, 3.0, "concentrate"),
    (9004, 75, 0.85, 5.0, "competitor-x"),
]


def start_mocks():
    for port, lat, acc, price, name in MOCKS:
        srv = serve(port, latency_ms=lat, tok_ms=1, fail_rate=0.01, accuracy=acc, price_per_mtok=price, name=name)
        threading.Thread(target=srv.serve_forever, daemon=True).start()


if __name__ == "__main__":
    os.environ.setdefault("MOCK_KEY", "mock")
    out = ROOT / "results" / "dryrun"
    shutil.rmtree(out, ignore_errors=True)
    start_mocks()
    sc = ["--systems-config", "configs/dryrun/systems.yaml", "--out", str(out)]
    main(sc + ["list"])
    main(sc + ["track-a", "--config", "configs/dryrun/track_a.yaml"])
    main(sc + ["track-b", "--config", "configs/dryrun/track_b.yaml"])
    main(sc + ["load", "--config", "configs/dryrun/load.yaml"])
    main(["report", str(out), "--report-dir", "reports/dryrun"])
    print("\nDry run complete. Open reports/dryrun/REPORT.md")
