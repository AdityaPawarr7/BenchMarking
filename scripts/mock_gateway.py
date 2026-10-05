"""A fake OpenAI-compatible gateway for dry runs and tests. No network, no cost.

    python scripts/mock_gateway.py --port 9001 --latency-ms 40 --fail-rate 0.0

It answers math/multiple-choice sample prompts correctly with probability --accuracy,
streams tokens with a fixed per-token delay, and reports usage.cost.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def make_handler(latency_ms: float, tok_ms: float, fail_rate: float, accuracy: float, price_per_mtok: float, name: str,
                 known_models=None, report_cost: bool = True):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # quiet
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if known_models is not None and body.get("model") not in known_models:
                self._json(404, {"error": {"code": "model_not_found", "type": "invalid_request_error",
                                           "message": f"The model '{body.get('model')}' does not exist"}})
                return
            if random.random() < fail_rate:
                self._json(503, {"error": {"message": "mock upstream unavailable"}})
                return
            time.sleep(latency_ms / 1000)
            prompt = " ".join(m.get("content", "") for m in body.get("messages", []))
            text = _answer(prompt, accuracy)
            words = text.split(" ")
            max_tokens = int(body.get("max_tokens", 64))
            if len(words) < max_tokens and "essay" in prompt:
                words = (words * (max_tokens // max(1, len(words)) + 1))[:max_tokens]
            in_tok = max(1, len(prompt) // 4)
            out_tok = len(words)
            usage = {"prompt_tokens": in_tok, "completion_tokens": out_tok, "total_tokens": in_tok + out_tok}
            if report_cost:
                usage["cost"] = (in_tok + out_tok) * price_per_mtok / 1e6
            model = body.get("model", "mock")
            served = f"{name}/{model}" if model in ("auto", "router", "openrouter/auto") else model
            if not body.get("stream"):
                self._json(200, {"id": "mock", "model": served, "usage": usage,
                                 "choices": [{"index": 0, "message": {"role": "assistant", "content": " ".join(words)}}]})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for i, w in enumerate(words):
                self._chunk({"model": served, "choices": [{"index": 0, "delta": {"content": (" " if i else "") + w}}]})
                time.sleep(tok_ms / 1000)
            self._chunk({"model": served, "choices": [], "usage": usage})
            self._raw(b"data: [DONE]\n\n")
            self._raw(b"", end=True)

        def _chunk(self, obj):
            self._raw(f"data: {json.dumps(obj)}\n\n".encode())

        def _raw(self, data: bytes, end: bool = False):
            self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
            if end:
                self.wfile.flush()

        def _json(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return H


def _answer(prompt: str, accuracy: float) -> str:
    right = random.random() < accuracy
    m = re.search(r"\[expected:(.+?)\]", prompt)
    if m:
        ans = m.group(1).strip()
        return f"The answer is {ans}. #### {ans}" if right else "The answer is 0. #### 0"
    return "Networks move packets between hosts using routing tables and protocols"


def serve(port: int, **kw) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(**kw))
    srv.daemon_threads = True
    return srv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9001)
    ap.add_argument("--latency-ms", type=float, default=40)
    ap.add_argument("--tok-ms", type=float, default=2)
    ap.add_argument("--fail-rate", type=float, default=0.0)
    ap.add_argument("--accuracy", type=float, default=0.8)
    ap.add_argument("--price-per-mtok", type=float, default=1.0)
    ap.add_argument("--name", default="mock")
    a = ap.parse_args()
    print(f"mock gateway on http://127.0.0.1:{a.port}/v1")
    serve(a.port, latency_ms=a.latency_ms, tok_ms=a.tok_ms, fail_rate=a.fail_rate, accuracy=a.accuracy,
          price_per_mtok=a.price_per_mtok, name=a.name).serve_forever()
