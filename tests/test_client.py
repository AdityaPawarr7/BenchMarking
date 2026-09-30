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
