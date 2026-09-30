"""Append-only JSONL results store."""
from __future__ import annotations

import json
import platform
import socket
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


class ResultWriter:
    def __init__(self, path: str | Path, run_meta: dict[str, Any] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")
        self.run_meta = {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "run_started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **(run_meta or {}),
        }

    def write(self, record: dict[str, Any]) -> None:
        self._f.write(json.dumps({**self.run_meta, **record}, default=str) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_results(paths: Iterable[str | Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        p = Path(p)
        files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
        for fp in files:
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    df = pd.json_normalize(rows, max_level=1)
    return df
