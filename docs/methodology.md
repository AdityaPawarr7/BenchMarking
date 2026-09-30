# Methodology (summary)

The full framework lives in the capstone doc "LLM Gateway Benchmark Framework". This file is the
short version the code follows.

## Hypotheses (pre-register before running)

1. **H1 Overhead:** Concentrate adds no more p50/p99 TTFT over a direct provider call than the fastest competing gateway.
2. **H2 Cost:** For the same model and tokens, Concentrate's billed cost is at or below the cheapest gateway.
3. **H3 Routing efficiency:** Concentrate reaches >= 95% of the best single model's accuracy at lower cost than competing routers.
4. **H4 Reliability:** Under provider failures and 429s, Concentrate's success rate and failover time match or beat gateways with fallbacks.
5. **H5 Scale:** Throughput and tail latency hold from 1 to 500+ req/s.

## Protocol

| Step | Setting |
|---|---|
| Region | Primary us-east-1, repeat in eu-west-1 |
| Schedule | 3 times of day x 5+ days, systems interleaved each round |
| Track A sample | 1,000 requests per system x shape x stream mode |
| Track B sample | 500-1,000 items per workload |
| Warm-up | 50 requests per system, discarded |
| Caching | Off (nonce per prompt); caching tested separately |
| Stats | Bootstrap 95% CIs; Mann-Whitney (latency); McNemar (paired accuracy); Holm correction |
| Scoring | Blind LLM judge, fixed judge model, 5% human audit |

## Composite weights (configs/scorecard.yaml)

Efficiency 35%, speed 25%, cost 20%, reliability 15%, features 5%. The ranking must be reported
under the alternative weightings too.
