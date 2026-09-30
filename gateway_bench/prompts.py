"""Synthetic prompts of a target size for Track A and load tests."""
from __future__ import annotations

import random
import uuid

_WORDS = (
    "system latency gateway router token model provider request response stream cache region "
    "throughput budget quality benchmark metric sample report cost price invoice network client "
    "server queue retry fallback error success window context summary answer question data"
).split()


def filler_text(approx_tokens: int, rng: random.Random) -> str:
    # ~0.75 words per token for English-like text
    n_words = max(1, int(approx_tokens * 0.75))
    return " ".join(rng.choice(_WORDS) for _ in range(n_words))


def shape_messages(input_tokens: int, output_tokens: int, *, nonce: bool = True, seed: int | None = None):
    rng = random.Random(seed)
    tag = f"[run {uuid.uuid4().hex[:12]}] " if nonce else ""
    body = filler_text(max(1, input_tokens - 40), rng)
    words = max(1, int(output_tokens * 0.75))
    return [
        {"role": "system", "content": "You are a benchmarking assistant. Follow length instructions exactly."},
        {
            "role": "user",
            "content": (
                f"{tag}Below is reference text. Ignore its meaning.\n\n{body}\n\n"
                f"Now write a continuous essay about computer networks of about {words} words. "
                "Do not stop early."
            ),
        },
    ]
