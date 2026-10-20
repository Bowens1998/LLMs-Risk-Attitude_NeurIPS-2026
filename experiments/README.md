# Data collection

Code that produces the trial data of the benchmark: the Phase-1 (reliability) LLM runner,
the browser interfaces used for Phase-2 (attitude estimation) LLM trials and for the human
study, and the scoring that turns Phase-2 trials into (B_C, R_D) pairs. No data are included;
the released data are on Hugging Face (`python data/download_data.py`).

| Path | Purpose |
|---|---|
| `runner/ic_runner.py` | Phase 1: three frozen scenarios per task, each replayed N times to one model (OpenRouter or any OpenAI-compatible endpoint, incl. Hugging Face Inference Endpoints; chat or raw completions) |
| `runner/replay_check.py` | offline check (no API): feeds the recorded responses of the published Base/Instruct runs back through the runner and compares B_C, R_D (CTD, DNC) and DNC trajectories |
| `config/phase1_frozen_scenarios.json` | the locked Phase-1 scenarios (3 per task) with SHA-256 per scenario |
| `config/frozen_ordinalization_map.json` | benchmark-v1 frozen R_D centers for DNC (Strategy Index) and FIP (high-risk share) |
| `scoring/rd_discretization.py` | Phase 2: B_C extraction, raw decision (SI / H share / ESI), R_D by nearest frozen center; optional pooled K-means refit |
| `web/llm/Experiment_{DNC,CTD,FIP}/` | Phase 2: browser task environments that query models through OpenRouter (single run, auto-run and parallel headless modes) |
| `web/human/` | human-participant portal and the three task pages (HTML/JS only) |

```bash
pip install -r experiments/requirements.txt     # requests, numpy (+ scikit-learn for --refit)
```

## Tasks and protocol

* **DNC** (drone navigation, 10×20 grid, ≤100 steps): per-step action + belief, then a
  post-trial danger rating (B_C). R_D from the Strategy Index on drift steps.
* **CTD** (clinical triage, ≤14 ticks): per-tick risk score `BC.ctx` and an optional ESI
  finalization; B_C is the last score, R_D the finalized ESI (1 = most urgent).
* **FIP** (portfolio, three assets): one market-risk estimate (B_C) and an L/M/H allocation;
  R_D from the high-risk share H/(L+M+H).

Decoding: temperature 0.7 (the runner default). Phase-1 condition parameters (wind/drift/walls,
prevalence/noise/volatility, regime/volatility/correlation) are the `*_PRESETS` in
`runner/ic_runner.py` and equal the paper's appendix tables; Phase-2 parameters are in each
`web/llm/Experiment_*/js/config.json`.

## Phase 1 — reliability (frozen scenarios × 30 repetitions)

API keys are read from environment variables only:

| Provider | Variables |
|---|---|
| `--provider openrouter` (default) | `OPENROUTER_API_KEY` (`OPENROUTER_KEY` is also accepted) |
| `--provider huggingface` | `HF_TOKEN`, and `HF_ENDPOINT_URL` (or `--base-url`) for a dedicated Inference Endpoint |

Always start with `--dry-run` (validates the scenario bundle and prints the plan; no requests):

```bash
python experiments/runner/ic_runner.py --exp all --n 30 --seed 42 \
    --scenario-bundle experiments/config/phase1_frozen_scenarios.json \
    --model qwen/qwen3-max --dry-run
```

Settings of the published matched Base/Instruct runs (one call per checkpoint; Qwen
endpoints served chat completions, Gemma endpoints raw completions):

```bash
export HF_TOKEN=...  HF_ENDPOINT_URL=https://<your-endpoint>
python experiments/runner/ic_runner.py --provider huggingface \
    --api-mode chat --model Qwen/Qwen3-30B-A3B-Base --output-model-name qwen3_30b_a3b_base \
    --exp all --levels 1,2,3 --n 30 --seed 42 \
    --scenario-bundle experiments/config/phase1_frozen_scenarios.json \
    --organized-output --run-id <RUN_ID> --checkpoint-every 10 \
    --temperature 0.7 --top-p 1.0 --max-tokens 512 --timeout 180 --api-retries 8 \
    --reruns 0 --max-extra-attempts 30 --invalidate-on-api-error
# Qwen/Qwen3-30B-A3B-Instruct-2507        --output-model-name qwen3_30b_a3b_instruct_2507
# google/gemma-4-E4B / google/gemma-4-E4B-it: --api-mode completions --completion-assistant-cue
#     --output-model-name gemma_4_e4b_base / gemma_4_e4b_it
```

`--max-extra-attempts 30` gives each scenario at most 30 + 30 attempts (≤180 per
checkpoint–task cell); invalid responses are replaced only within that budget and count as
completion failures. With a locked bundle the manipulation check (target B_C 15/50/85 ± 30) is
reported but never regenerates a scenario. Without `--scenario-bundle` the runner draws a new
frozen scenario per condition from `--seed` and regenerates a condition whose mean B_C misses
its target (up to `--reruns` rounds). Interrupted runs resume from `_checkpoints/` with
`--resume` and the same `--run-id`. For OpenRouter models use their OpenRouter ids, e.g.
`openai/gpt-5.2`, `x-ai/grok-4`, `anthropic/claude-sonnet-4.5`, `deepseek/deepseek-v3.2`,
`google/gemini-3-pro-preview`, `qwen/qwen3-max`.

**Outputs** (default `outputs/experiments/`, or `--out DIR`): with `--organized-output`,
`<out>/<TASK>/<model>/<run-id>/<TASK>_<model>_N<n>.json` and `..._meta.json` — the same
layout and schema as the per-run folders under `FrontierModel_reliability_N90/` in the dataset.
The primary file is a flat array with the eight fields of `Reliability_<TASK>_merged.json`
(`experiment, model, model_key, level, trial_index, level_trial_index, contextual_belief,
risk_decision`); the merged files are the concatenation of such per-run arrays. The sidecar
holds the seed, decoding settings, attempted trials per scenario (`configs[].attempts`, used for
completion yield), the frozen scenarios and all raw model responses.

**Offline check.** `python experiments/runner/replay_check.py` (needs the dataset with its
`*_meta.json` sidecars) replays every recorded response of the twelve Base/Instruct runs and
reports, per run, how many B_C, R_D (CTD, DNC) and DNC trajectories are reproduced and whether
the scenarios match `phase1_frozen_scenarios.json`.

## Phase 2 — attitude estimation (100 trials per model and task)

The browser pages draw a fresh environment for every trial and call OpenRouter directly from
the browser. Serve them over HTTP (they load `js/config.json`):

```bash
python -m http.server 8000 --directory experiments/web/llm
# open http://localhost:8000/Experiment_DNC/  (…/Experiment_CTD/, …/Experiment_FIP/)
```

1. **API Settings**: paste an OpenRouter key (kept only in the browser's `localStorage`; never
   written to files), choose the model id, temperature 0.7, then *Test* and *Save*.
2. Run 100 trials: DNC — enable API use in *API Settings*, *LLM Mode*, Trials/Block = 100,
   *Start*, then *Auto Run (API)*; CTD and FIP — the *Parallel Execution (Headless)* panel
   (groups × trials per group = 100), *Start Parallel Run*.
3. *Download* the JSON log (CTD/FIP: *Download Combined Results*).

The downloaded logs contain the per-trial objects stored in the dataset's `*_main_N100/<TASK>/`
files (`{"experiment", "model", "n_trials", "trials": [...]}`): DNC trials with `steps`
(action, position, `drift_row`, belief) and the survey `context_belief`; CTD trials with
`steps[].BC_updates` and `finalized.ESI`; FIP trials with `prices`, `pct` and `report`
(`contextual.risk`, `alloc`). Score them (or the dataset files) with

```bash
python experiments/scoring/rd_discretization.py data/hf/FrontierModel_main_N100/*/*.json
python experiments/scoring/rd_discretization.py --refit my_dnc_run.json   # also show pooled K-means centers
```

which writes `outputs/experiments/phase2_scored.csv` (B_C, raw decision, R_D, out-of-support
flag) and prints R_D counts per model. On the released frontier files it reproduces the paper's
category usage (11/18 cells use all five categories). Benchmark v1 keeps the frozen centers;
refitting K-means on another pool changes the centers and therefore some categories.

## Human-participant interface

`web/human/index.html` is the portal (consent, rules, links to the three tasks; each task page has a
guided tour and five trials). Logs stay in the participant's browser
until all three tasks are done, then one payload is posted to your webhook. See
`web/human/deployment_guide.md` (replace `YOUR_WEBHOOK_URL_HERE` and
`YOUR_PROLIFIC_COMPLETION_CODE`). The study ran under IRB approval; reuse requires your own
ethics approval and consent text.

## Cost, time and reproducibility caveats

* Calls per trial: FIP 1; CTD up to 14 (one per tick); DNC up to 101 (one per step plus the
  survey). A 90-trial Phase-1 DNC run can need ~2,000–9,000 requests (more with retries and
  extra attempts) and take several hours; Phase-2 runs need roughly 10/9 of that per task.
* Runs cost money on commercial APIs and dedicated endpoints bill while running; check
  quotas first and use `--dry-run`.
* Closed models behind APIs change without notice and provider-side sampling is not seeded
  (only the local scenario generator is), so reruns are not bitwise reproducible; frozen
  scenarios, prompts, parsing and scoring are. Provider defaults (e.g. top-p, penalties) may
  also differ across providers; the runner sends `top_p` explicitly (default 1.0).
