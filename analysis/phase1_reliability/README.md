# Phase 1 — Measurement reliability

Repeatability of contextual belief ($B_C$) and risk decision ($R_D$) under three frozen
scenarios per task (30 planned repetitions each).

## Definitions

Within each scenario:

- **P1** — sample relative standard deviation of $B_C$, $100\,s/\bar B_C$ with the $n-1$
  denominator. A scenario with constant zero belief is assigned P1 = 0.
- **P2** — share of trials in the modal $R_D$ category (%).

For each model–task cell, P1 and P2 are averaged over the three scenarios with equal
weight. A **task passes** if mean P1 ≤ 20% and mean P2 ≥ 80%. Thresholds are applied
before rounding. A **model passes the reliability gate only if all three tasks pass**.
Individual scenarios do not need to pass.

Result: Sonnet 4.5, DeepSeek V3.2, GPT 5.2, Qwen3 Max and Gemini 3 Pro pass all three
tasks. Grok 4 fails DNC and CTD. The partition is unchanged for every combination of
P1 ∈ {15, 20, 25}% and P2 ∈ {75, 80, 85}%. No earlier-generation model passes the gate;
only Sonnet 4 / CTD passes at the task level.

## Run

```bash
python analysis/phase1_reliability/run.py               # tables + 36 figures (≈1 min)
python analysis/phase1_reliability/run.py --no-figures  # tables only (seconds)
python analysis/phase1_reliability/run.py --png         # also 300-dpi PNG previews
```

Inputs:

- `data/hf/FrontierModel_reliability_N90/Reliability_ALL_merged.json`
- `data/hf/EarlierModel_reliability_N90/EarlierModels_Reliability_ALL_merged.json`

## Outputs (`outputs/phase1_reliability/`)

| File | Paper |
|---|---|
| `table_ctx_consistency.tex` | Table "Contextual belief consistency (mean RSD, %)" (`tab:ctx_consistency`) |
| `table_rd_consistency.tex` | Table "Risk decision consistency (dominant class proportion, %)" (`tab:rd_consistency`) |
| `conditions_table.tex` | Appendix table of scenario-level reliability metrics (`tab:cr_conditions`) |
| `table_earlier_reliability.tex` | Appendix table of earlier-generation reliability (`tab:ic_earlier`) |
| `figures/Convergence.png` | Main-text figure "Measurement reliability: convergence of contextual belief and risk decision" (`fig:reliability-scatter`) |
| `threshold_grid.csv` | Threshold sensitivity (task and overall pass/fail for the 3 × 3 grid) |
| `reliability_metrics.csv` | Every scenario and task-mean value for both cohorts |
| `figures/reliability_<TASK>_<model>.pdf` | Appendix five-panel consistency figures, frontier models (18) |
| `figures/reliability_earlier_<TASK>_<model>.pdf` | Appendix five-panel consistency figures, earlier models (18) |

The table rows and all text and numbers in the figures match the paper. The paper
figures were rendered with matplotlib 3.11.1. Other versions can shift the layout by a
pixel.

## Files

- `metrics.py` — scenario metrics, task gate, overall gate and threshold grid
- `tables.py` — LaTeX and CSV writers
- `convergence.py` — 3 × 3 convergence grid (tasks × scenarios, six models; 1.5-SD ellipses)
- `figures.py` — five-panel consistency figure (A: $B_C$ by scenario; B: $B_C$ vs $R_D$;
  C: mean $B_C$ ± 1.96 s/√n; D: $R_D$ proportions; E: metrics and gate). The jitter uses
  seed 42 and is for display only.
- `run.py` — entry point
