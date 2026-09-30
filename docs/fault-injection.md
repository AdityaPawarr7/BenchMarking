# Fault injection (H4 reliability)

Goal: measure how each gateway behaves when its primary upstream fails, and how long failover takes.

Hosted gateways call providers from their own servers, so you cannot put a proxy between them and
the provider. Use one of these instead:

1. **BYOK with a broken key / bad endpoint** (hosted gateways that support BYOK + fallbacks):
   configure primary = provider with an invalid or rate-limited key, fallback = a working provider.
   Every request exercises failover; compare TTFT to a normal run to get the failover cost.
2. **Self-hosted proxies (LiteLLM, Bifrost, TensorZero, Kong, RouteLLM):** point the proxy's primary
   upstream at [Toxiproxy](https://github.com/Shopify/toxiproxy) and inject faults live:

```bash
toxiproxy-server &
toxiproxy-cli create -l 127.0.0.1:18443 -u api.openai.com:443 openai
# 100% failure for 60 s
toxiproxy-cli toxic add -t timeout -a timeout=0 openai
# 5 s added latency
toxiproxy-cli toxic add -t latency -a latency=5000 openai
```

3. **Mock upstream:** `python scripts/mock_gateway.py --fail-rate 0.5` as the primary upstream.

Run `gbench track-a --shapes tiny --requests 200` during the fault window and record the window's
start/end times. Report success rate, p50/p99 TTFT inside the window, and time from fault start to
the first successful response.
