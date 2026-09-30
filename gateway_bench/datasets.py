"""Workload loading and preparation.

A workload is a JSONL file; each line:
  {"id": str, "prompt": str, "answer": any, "scorer": "numeric|mc|exact|contains|code|judge",
   "system": optional system prompt, "tests": optional (code scorer)}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def load_workload(path: str | Path, max_items: int | None = None) -> list[dict[str, Any]]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if max_items and len(items) >= max_items:
                break
    for i, it in enumerate(items):
        it.setdefault("id", str(i))
        it.setdefault("scorer", "judge")
    return items


def _write(path: Path, rows: Iterator[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
            n += 1
    return n


def prepare_all(out_dir: str | Path = "data/prepared", limit: int | None = None, seed: int = 0) -> dict[str, int]:
    """Download public benchmarks via Hugging Face `datasets` and convert them.

    Requires: pip install -e '.[data]'
    """
    from datasets import load_dataset  # optional dependency

    out = Path(out_dir)
    counts: dict[str, int] = {}

    def take(ds):
        ds = ds.shuffle(seed=seed)
        return ds.select(range(min(limit, len(ds)))) if limit else ds

    # GSM8K: grade-school math, numeric answer after '####'
    gsm = take(load_dataset("openai/gsm8k", "main", split="test"))
    counts["gsm8k"] = _write(out / "gsm8k.jsonl", (
        {"id": f"gsm8k-{i}", "prompt": r["question"] + "\n\nThink step by step, then give the final answer as '#### <number>'.",
         "answer": r["answer"].split("####")[-1].strip(), "scorer": "numeric"}
        for i, r in enumerate(gsm)))

    # MMLU-Pro: 10-way multiple choice
    mp = take(load_dataset("TIGER-Lab/MMLU-Pro", split="test"))
    letters = "ABCDEFGHIJ"

    def mmlu_rows():
        for i, r in enumerate(mp):
            opts = "\n".join(f"{letters[j]}. {o}" for j, o in enumerate(r["options"]))
            yield {"id": f"mmlupro-{r.get('question_id', i)}",
                   "prompt": f"{r['question']}\n\n{opts}\n\nAnswer with the letter of the correct option, as 'Answer: X'.",
                   "answer": r["answer"], "scorer": "mc", "category": r.get("category")}
    counts["mmlu_pro"] = _write(out / "mmlu_pro.jsonl", mmlu_rows())

    # MATH-500: competition math, judged (answers are LaTeX expressions)
    m5 = take(load_dataset("HuggingFaceH4/MATH-500", split="test"))
    counts["math500"] = _write(out / "math500.jsonl", (
        {"id": f"math500-{i}", "prompt": r["problem"] + "\n\nPut the final answer in \\boxed{}.",
         "answer": r["answer"], "scorer": "judge"}
        for i, r in enumerate(m5)))

    return counts
