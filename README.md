# gateway-bench

Benchmark harness for comparing **Concentrate's router** with other LLM gateways and routers
(Ramp Router, OpenRouter, Martian, Not Diamond, Requesty, Unify, RouteLLM, Portkey, LiteLLM,
Helicone, Cloudflare, Vercel, Kong, Bifrost, TensorZero, Azure and Bedrock routers) on
**cost, speed, quality per dollar and reliability**.

It implements the capstone methodology: two separate tracks, so a routing win can never hide
a proxy loss (or the reverse).

| Track | Question | Holds fixed | Key outputs |
|---|---|---|---|
| **A: gateway overhead** | What does the gateway itself cost in time and money? | Model + upstream provider | TTFT overhead vs direct (p50/p99, bootstrap CI), throughput, effective markup, success rate |
| **B: routing quality** | Does the router get more correct answers per dollar? | Prompts + scoring | Accuracy (CI), cost per correct answer, quality retained vs best model, cost saved vs strongest, Pareto frontier, AIQ, McNemar tests |
| **Load** | Does it hold up at scale? | Prompt shape | Achieved RPS, p99 TTFT, error rate per step, max sustained RPS |

Everything ends in a composite **scorecard** with pre-registered weights and a sensitivity check.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

# 1. Prove the pipeline works with local mock gateways (no keys, no cost, ~1 min)
python scripts/dryrun.py          # -> reports/dryrun/REPORT.md + charts
pytest

# 2. Configure real systems
cp .env.example .env              # add one API key per system
$EDITOR configs/systems.yaml      # confirm base URLs + model IDs (lines marked TODO/verify)
gbench list                       # shows which systems have keys set

# 3. Pilot (small, cheap), then full runs
gbench track-a --requests 20 --shapes tiny,short
pip install -e '.[data]' && gbench prepare-data
gbench track-b --workloads sample
gbench track-a
gbench track-b
gbench load --systems concentrate,openrouter,ramp-router

# 4. Analyse
gbench report results --report-dir reports/$(date +%Y%m%d)
```

## Repo layout

```
configs/
  systems.yaml      every gateway/router: base_url, key env var, tracks, model aliases, router_model
  track_a.yaml      prompt shapes, streaming modes, requests per cell, baseline system
  track_b.yaml      workloads, routers, single-model baselines (candidate pool), LLM judge
  load.yaml         arrival-rate steps, step duration, SLO
  prices.yaml       fallback list prices + known markups (used only when cost isn't reported)
  scorecard.yaml    composite weights, sensitivity alternatives, feature-parity matrix
  dryrun/           mock configs used by scripts/dryrun.py
gateway_bench/
  client.py         async OpenAI-compatible client; records TTFT, e2e, tokens, served model, billed cost
  track_a.py        interleaved, randomized overhead runs with warm-up and cache-busting nonces
  track_b.py        routing runs + scoring (numeric, multiple choice, exact, code, blind LLM judge)
  load.py           open-loop Poisson load generator
  analysis.py       bootstrap CIs, Mann-Whitney, McNemar, Holm, Pareto, AIQ, scorecard
  report.py         CSV tables, latency CDFs, cost-quality frontiers, REPORT.md
  datasets.py       workload JSONL format + GSM8K / MMLU-Pro / MATH-500 preparation
scripts/
  mock_gateway.py   fake OpenAI-compatible server for tests and dry runs
  dryrun.py         end-to-end run against 4 mock gateways
data/sample/        tiny smoke-test workloads
docs/               methodology notes and fault-injection guide
results/            raw JSONL (git-ignored)
```

## Adding a system

Add an entry to `configs/systems.yaml`. Any OpenAI-compatible `/chat/completions` endpoint works.

```yaml
- name: my-router
  kind: router                  # baseline | gateway | router
  base_url: https://api.example.com/v1
  api_key_env: MY_ROUTER_API_KEY
  tracks: [A, B]
  models: {small: openai/gpt-4.1-mini}   # Track A alias -> this system's id for the SAME upstream model
  router_model: auto            # model string that turns on its routing (Track B)
  headers: {x-extra: "${SOME_ENV}"}
  extra_body: {usage: {include: true}}
```

To test several router settings (a cost/quality curve for AIQ), add one entry per setting and name
them `concentrate@cheap`, `concentrate@balanced`, ... ; the report groups them by the prefix.

## Fairness rules the harness enforces (and ones you must)

Enforced in code: same client and parameters for every system, `temperature 0` + fixed seed,
randomized interleaving of systems in each round, discarded warm-up requests, a unique nonce per
Track A prompt so no cache can answer, served-model logging (`served_model_mismatch` in the report),
open-loop arrivals in load tests, no client retries.

Your responsibility: run from the same region (repeat from a second one), at 3 times of day over
5+ days, pin every Track A system to the same upstream provider, freeze Concentrate's router config
before Track B, and reconcile computed cost with each vendor's invoice. See `docs/methodology.md`.

## Result record

Every call is one JSON line in `results/*.jsonl` with: `system`, `model_requested`, `model_served`,
`stream`, `status`, `ok`, `error`, `t_start`, `ttft_ms`, `e2e_ms`, `input_tokens`, `output_tokens`,
`output_tps`, `cost_reported`, `meta` (shape / workload / item / label), and in Track B `correct`,
`cost_usd`, `cost_source` (`reported` or `list_price`).

## Safety

- `.env` and `results/` are git-ignored. Don't commit keys or raw outputs of private data.
- Code-execution scoring runs model-written code; it is disabled unless `GBENCH_ALLOW_CODE_EXEC=1`,
  and should only be enabled inside a throwaway container.
- Check each vendor's terms on benchmarking and publishing results before publishing names.
