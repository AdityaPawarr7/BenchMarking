import asyncio
import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mock_gateway import serve  # noqa: E402

from gateway_bench.client import chat, make_client  # noqa: E402
from gateway_bench.config import System  # noqa: E402

PORT = 9107


def setup_module(_):
    srv = serve(PORT, latency_ms=30, tok_ms=1, fail_rate=0.0, accuracy=1.0, price_per_mtok=2.0, name="mock")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    os.environ["MOCK_KEY"] = "x"


def _sys():
    return System(name="mock", kind="gateway", base_url=f"http://127.0.0.1:{PORT}/v1", api_key_env="MOCK_KEY")


def _run(stream):
    async def go():
        async with make_client() as c:
            return await chat(c, _sys(), "m", [{"role": "user", "content": "q [expected:7]"}], stream=stream)
    return asyncio.run(go())


def test_stream_records_ttft_tokens_cost():
    r = _run(True)
    assert r.ok and r.status == 200
    assert r.ttft_ms is not None and r.e2e_ms >= r.ttft_ms >= 30
    assert r.output_tokens and r.cost_reported > 0
    assert "#### 7" in r.text


def test_non_stream():
    r = _run(False)
    assert r.ok and r.ttft_ms == r.e2e_ms and "#### 7" in r.text


def test_error_is_captured_not_raised():
    s = System(name="dead", kind="gateway", base_url="http://127.0.0.1:1/v1", api_key_env="MOCK_KEY", timeout_s=2)

    async def go():
        async with make_client() as c:
            return await chat(c, s, "m", [{"role": "user", "content": "hi"}])
    r = asyncio.run(go())
    assert not r.ok and r.error


def test_hard_deadline_on_trickling_response():
    """A server that keeps sending bytes slowly must not hang the run (per-read timeouts never fire)."""
    import socket
    import time as _t
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Trickle(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                for _ in range(100):
                    self.wfile.write(b": keep-alive\n\n"); self.wfile.flush(); _t.sleep(0.2)
            except (BrokenPipeError, ConnectionResetError, socket.error):
                pass

    srv = ThreadingHTTPServer(("127.0.0.1", 9441), Trickle)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    s = System(name="slow", kind="gateway", base_url="http://127.0.0.1:9441/v1", api_key_env="MOCK_KEY", timeout_s=1.5)

    async def go():
        async with make_client() as c:
            return await chat(c, s, "m", [{"role": "user", "content": "hi"}], stream=True)
    t0 = _t.time()
    r = asyncio.run(go())
    srv.shutdown()
    assert not r.ok and r.error.startswith("timeout")
    assert _t.time() - t0 < 5


def test_credit_guard_trips_on_402s():
    from gateway_bench.client import CallResult, CreditGuard, OutOfCredits
    g = CreditGuard(limit=3)
    for i in range(7):
        g.record(CallResult(request_id=str(i), system="openrouter", model_requested="m", ok=(i % 2 == 0),
                            status=200 if i % 2 == 0 else 402))
    assert g.tripped and "openrouter" in g.tripped
    try:
        g.check()
        raise AssertionError("should raise")
    except OutOfCredits:
        pass
