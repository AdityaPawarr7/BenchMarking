"""Scorers for Track B. Each returns True/False (or None when it cannot score)."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from typing import Any

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()).strip(" .")


def score_exact(output: str, answer: Any) -> bool:
    return _norm(output) == _norm(str(answer))


def score_contains(output: str, answer: Any) -> bool:
    return _norm(str(answer)) in _norm(output)


def extract_final_number(text: str) -> float | None:
    # prefer "#### 42" / "answer: 42" / \boxed{42}, else the last number in the text
    for pat in (r"####\s*(-?[\d,]*\.?\d+)", r"\\boxed\{\s*(-?[\d,]*\.?\d+)\s*\}", r"answer\s*(?:is|:)\s*\$?(-?[\d,]*\.?\d+)"):
        m = re.findall(pat, text, flags=re.I)
        if m:
            return _to_float(m[-1])
    nums = _NUM_RE.findall(text)
    return _to_float(nums[-1]) if nums else None


def _to_float(s: str) -> float | None:
    try:
        return float(s.replace(",", "").rstrip("."))
    except ValueError:
        return None


def score_numeric(output: str, answer: Any, tol: float = 1e-6) -> bool:
    got = extract_final_number(output)
    want = _to_float(str(answer)) if not isinstance(answer, (int, float)) else float(answer)
    return got is not None and want is not None and abs(got - want) <= tol * max(1.0, abs(want))


def extract_choice(text: str) -> str | None:
    for pat in (r"answer\s*(?:is|:)?\s*\(?([A-J])\)?\b", r"^\s*\(?([A-J])\)?[\s.):]", r"\b([A-J])\b(?!.*\b[A-J]\b)"):
        m = re.search(pat, text, flags=re.I | re.S | re.M)
        if m:
            return m.group(1).upper()
    return None


def score_mc(output: str, answer: Any) -> bool:
    return extract_choice(output) == str(answer).strip().upper()


def score_code(output: str, tests: str, timeout_s: int = 15) -> bool:
    """Run generated code + tests in a subprocess.

    WARNING: this executes model-written code. Run Track B coding workloads inside a
    throwaway container or VM with no credentials mounted.
    """
    if os.environ.get("GBENCH_ALLOW_CODE_EXEC") != "1":
        raise RuntimeError("code scoring disabled; set GBENCH_ALLOW_CODE_EXEC=1 inside a sandbox")
    m = re.search(r"```(?:python)?\n(.*?)```", output, flags=re.S)
    code = m.group(1) if m else output
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code + "\n\n" + tests + "\n")
        path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout_s)
        return p.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(path)


JUDGE_PROMPT = """You are grading an AI assistant's answer. You do not know which system produced it.

Question:
{question}

Reference answer (may be partial):
{reference}

Candidate answer:
{candidate}

Is the candidate answer correct and complete with respect to the question and reference?
Reply with exactly one word: CORRECT or INCORRECT."""


def parse_judge(text: str) -> bool | None:
    t = text.strip().upper()
    if "INCORRECT" in t:
        return False
    if "CORRECT" in t:
        return True
    return None


def score_sync(scorer: str, output: str, item: dict[str, Any]) -> bool | None:
    ans = item.get("answer")
    if scorer == "exact":
        return score_exact(output, ans)
    if scorer == "contains":
        return score_contains(output, ans)
    if scorer == "numeric":
        return score_numeric(output, ans)
    if scorer == "mc":
        return score_mc(output, ans)
    if scorer == "code":
        return score_code(output, item.get("tests", ""))
    return None  # "judge" is async and handled by track_b
