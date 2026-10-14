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


GPQA_PROMPT = ("{q}\n\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n\n"
               "Think it through, then end with the letter of the correct option, as 'Answer: X'.")


def gpqa_rows(ds, seed: int = 0) -> Iterator[dict[str, Any]]:
    """GPQA stores the right answer first; shuffle the four options with a fixed seed per question."""
    import random
    for i, r in enumerate(ds):
        opts = [(r["Correct Answer"], True)] + [(r[f"Incorrect Answer {k}"], False) for k in (1, 2, 3)]
        rid = r.get("Record ID") or f"gpqa-{i}"
        random.Random(f"{seed}:{rid}").shuffle(opts)
        letter = "ABCD"[[ok for _, ok in opts].index(True)]
        yield {"id": f"gpqa-{rid}", "prompt": GPQA_PROMPT.format(q=r["Question"].strip(), a=opts[0][0].strip(),
               b=opts[1][0].strip(), c=opts[2][0].strip(), d=opts[3][0].strip()),
               "answer": letter, "scorer": "mc", "domain": r.get("High-level domain")}


DATASETS = ("gsm8k", "mmlu_pro", "math500", "gpqa_diamond")


def prepare_all(out_dir: str | Path = "data/prepared", limit: int | None = None, seed: int = 0,
                only: list[str] | None = None, log=print) -> dict[str, Any]:
    """Download public benchmarks via Hugging Face `datasets` and convert them.

    Requires: pip install -e '.[data]'. GPQA is gated: accept its terms on huggingface.co
    (Idavidrein/gpqa) and set HF_TOKEN.
    Returns {name: row count} or {name: "error: ..."} per dataset.
    """
    import os

    from datasets import load_dataset  # optional dependency

    out = Path(out_dir)
    counts: dict[str, Any] = {}
    want = set(only or DATASETS)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    def take(ds):
        ds = ds.shuffle(seed=seed)
        return ds.select(range(min(limit, len(ds)))) if limit else ds

    def step(name, fn):
        if name not in want:
            return
        try:
            counts[name] = fn()
            log(f"[data] {name}: {counts[name]} questions")
        except Exception as e:  # noqa: BLE001 - one gated/offline set must not block the others
            counts[name] = f"error: {e}"
            log(f"[data] {name} failed: {e}")

    # GSM8K: grade-school math, numeric answer after '####'
    def gsm8k():
        gsm = take(load_dataset("openai/gsm8k", "main", split="test"))
        return _write(out / "gsm8k.jsonl", (
            {"id": f"gsm8k-{i}", "prompt": r["question"] + "\n\nThink step by step, then give the final answer as '#### <number>'.",
             "answer": r["answer"].split("####")[-1].strip(), "scorer": "numeric"}
            for i, r in enumerate(gsm)))

    # MMLU-Pro: 10-way multiple choice
    def mmlu_pro():
        mp = take(load_dataset("TIGER-Lab/MMLU-Pro", split="test"))
        letters = "ABCDEFGHIJ"

        def rows():
            for i, r in enumerate(mp):
                opts = "\n".join(f"{letters[j]}. {o}" for j, o in enumerate(r["options"]))
                yield {"id": f"mmlupro-{r.get('question_id', i)}",
                       "prompt": f"{r['question']}\n\n{opts}\n\nAnswer with the letter of the correct option, as 'Answer: X'.",
                       "answer": r["answer"], "scorer": "mc", "category": r.get("category")}
        return _write(out / "mmlu_pro.jsonl", rows())

    # MATH-500: competition math, judged (answers are LaTeX expressions)
    def math500():
        m5 = take(load_dataset("HuggingFaceH4/MATH-500", split="test"))
        return _write(out / "math500.jsonl", (
            {"id": f"math500-{i}", "prompt": r["problem"] + "\n\nPut the final answer in \\boxed{}.",
             "answer": r["answer"], "scorer": "judge"}
            for i, r in enumerate(m5)))

    # GPQA Diamond: 198 graduate-level science questions, 4 options (gated on Hugging Face)
    def gpqa():
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", token=token)
        return _write(out / "gpqa_diamond.jsonl", gpqa_rows(take(ds), seed))

    step("gsm8k", gsm8k)
    step("mmlu_pro", mmlu_pro)
    step("math500", math500)
    step("gpqa_diamond", gpqa)
    return counts
