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

Needs Python 3.9+ (the macOS built-in `python3` works). Keep the virtualenv outside iCloud folders:

```bash
python3 -m venv ~/.venvs/gbench && source ~/.venvs/gbench/bin/activate
pip install --upgrade pip          # macOS ships pip 21, too old for editable installs
pip install -e '.[dev,ui]'

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

## Web UI (v1: Concentrate vs OpenRouter)

```bash
pip install -e '.[ui]'
gbench ui                         # open http://127.0.0.1:8765
```

- **Demo mode** runs against mock gateways on your machine (free, ~30 s), so you can see the
  whole flow before adding keys. Demo numbers are synthetic and labelled as such.
- **Live mode** calls the real services using `CONCENTRATE_API_KEY`, `OPENROUTER_API_KEY`
  (and optionally `OPENAI_API_KEY` for the direct-call baseline) from `.env`. Keys never leave
  the server process.
- Pick tests (speed & fees, routing quality, load), set sample sizes, see the estimated number of
  calls, then watch live progress. Results show a head-to-head scorecard (each gap marked
  *confirmed* or *not yet conclusive* by its statistical test), latency distributions, an
  accuracy-vs-cost chart with the models each router chose, and load-test tail latency.
- **Model picker:** choose the model for Speed & fees and Load from OpenRouter's live catalog
  (OpenAI, Anthropic, DeepSeek, Z.ai GLM), or type any `author/model` ID. **Served by** pins the
  upstream provider (e.g. Novita, DeepSeek, Z.ai) on both gateways so only the gateway differs,
  and the direct-call baseline goes to that same provider (`configs/providers.yaml`; add
  `DEEPSEEK_API_KEY`, `NOVITA_API_KEY`, `ZAI_API_KEY` as needed). Concentrate's pin format is a
  TODO in `configs/systems.yaml`; until it's set the UI warns that its upstream may differ.
- **Pre-run check:** before each test, one tiny request goes to each system with the exact model and
  settings. If Concentrate or OpenRouter rejects it (wrong model name, bad key, disabled upstream
  account), the run stops with the reason and a hint instead of failing every request. Failures
  during a run show in an **Errors** panel and mark the run "Done with errors".
- **Concentrate's model name:** if Concentrate names a model differently from OpenRouter, set
  "Concentrate's name for this model" (suggestions come from Concentrate's `/models` list if it has one).
- **Cost:** OpenRouter reports billed cost per request. Concentrate doesn't (yet), so its cost is
  estimated from OpenRouter's list price for the model it served and labelled as an estimate.
- **Grader:** open-ended questions can be graded through OpenRouter (pick a grader model), so no
  OpenAI key is needed. Direct-provider baselines and single-model reference points switch off
  automatically when their keys aren't set.
- **Fairness checks:** (1) enter what each dashboard charged for a run; the page compares it with
  list price for the tokens each side actually processed and prices the *same* work for both, so
  totals aren't skewed when one side did more requests. (2) When failure rates differ (e.g. one side
  ran out of credits), every speed metric switches to *matched requests*: only requests both
  gateways completed, paired by size, mode and round.
- **Metrics:** time to first token p50/p90/p99 (streaming only), head-to-head first token on the same
  requests, total response time p50/p99, tail consistency (p99÷p50), slow outliers, output speed,
  success rate, tokens counted per request (billing accounting), cost per 1M tokens, charged ÷ list
  price, and charged for identical work, plus routing accuracy and cost per correct answer.
- **Cheapest on both:** the model picker suggests the cheapest models both gateways offer.
- Every run is saved under `results/ui/<run-id>/` (raw JSONL + the exact configs used), so
  `gbench report results/ui/<run-id>` produces the same analysis as a Markdown report.

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
  ui_demo/          mock configs used by the UI demo mode
gateway_bench/
  client.py         async OpenAI-compatible client; records TTFT, e2e, tokens, served model, billed cost
  track_a.py        interleaved, randomized overhead runs with warm-up and cache-busting nonces
  track_b.py        routing runs + scoring (numeric, multiple choice, exact, code, blind LLM judge)
  load.py           open-loop Poisson load generator
  analysis.py       bootstrap CIs, Mann-Whitney, McNemar, Holm, Pareto, AIQ, scorecard
  report.py         CSV tables, latency CDFs, cost-quality frontiers, REPORT.md
  ui/               local web UI (FastAPI server + single-page front end)
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
