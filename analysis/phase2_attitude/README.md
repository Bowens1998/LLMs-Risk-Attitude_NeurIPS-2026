# Phase 2: risk-attitude analyses

Regenerates every Phase-2 (attitude-estimation) number, table and figure of the paper for the
six frontier models and the six earlier-generation models: ordered logistic regression (OLR)
fits, risk attitude bias (AUC), risk sensitivity (slope beta) with Wald tests and
Benjamini-Hochberg adjustment, trial-bootstrap intervals, Kendall's W, pairwise ordering
probabilities, rank tables, the frozen benchmark-v1 ordinalization and its k = 4/6 sensitivity.

## Inputs

All inputs come from the public dataset mirrored in `data/hf/` (run `python data/download_data.py`):

| Input | Used for |
|---|---|
| `FrontierModel_main_N100/{DNC,CTD,FIP}/*.json` | frontier-model fits, bootstrap, ordering, figures |
| `EarlierModel_main_N100/{DNC,CTD,FIP}/*.json` | earlier-model fits, rank table and bar charts |
| `reference_row_order.json` (this folder) | trial order of two cells for draw-for-draw bootstrap reproduction |

Variables per trial (one $B_C$ and one $R_D$; $x = B_C/100$):

| Task | $B_C$ | Raw decision | $R_D$ |
|---|---|---|---|
| DNC | `context_belief` (post-trial danger rating) | Strategy Index on drift steps | nearest frozen center |
| CTD | last `BC_updates[-1].ctx` | `finalized.ESI` | ESI (native 1-5) |
| FIP | `report.contextual.risk` | H/(L+M+H) of `report.alloc` | nearest frozen center |

The Strategy Index is `SI = logit(clip(VCR / (VCR + HPR + 0.001), 0.001, 0.999))`, where VCR and
HPR are the shares of UP/DOWN and RIGHT actions on steps with nonzero drift (all steps if none).
Earlier-model files carry a precomputed `risk_decision`, which is used as released.

## Method

* **OLR.** `logit P(Y <= k | x) = theta_k - beta x`, fitted per model-task cell with
  `statsmodels` `OrderedModel` (logit link, BFGS, up to 1,000 iterations).
  `E[Y | x] = sum_k k P(Y = k | x)` over the categories observed in the cell;
  AUC is its trapezoid integral over a 1,001-point grid on [0, 1].
  `AUC_S` (diagnostics) averages the same curve over the cell's own observed $B_C$ range.
* **Slope inference.** Two-sided Wald p for beta; Benjamini-Hochberg across the 18 frontier cells.
* **Trial bootstrap.** 1,000 resamples of the 100 trials of each cell, refit, percentile 95%
  intervals. Indices are drawn from `numpy.random.default_rng(20260724)` in the order
  draw -> task (DNC, CTD, FIP) -> model (DeepSeek V3.2, Gemini 3 Pro, GPT-5.2, Grok 4, Qwen3 Max,
  Sonnet 4.5); refits run in parallel without changing the draws.
* **Cross-task ordering** (five models passing the Phase-1 reliability gate; Grok 4 excluded):
  Kendall's W of the AUC rank matrix at the point estimates and for every joint bootstrap draw;
  pairwise probabilities P(AUC_a > AUC_b) from 100,000 independent draws of each cell's bootstrap
  AUC distribution (`default_rng(2741)`, ties count one half). Models are ordered by the number of
  pairwise probabilities above 0.5, with summed probabilities breaking ties; the consensus order
  applies the same rule to probabilities averaged over the three tasks.
* **Frozen ordinalization (benchmark v1).** DNC and FIP raw decisions are assigned to the nearest
  of five fixed centers (`ordinalization.FROZEN_CENTERS`, also written to
  `frozen_ordinalization_v1.json`). Exact ties go to the smaller label; values outside the v1
  reference range map to the nearest extreme category and are flagged `out_of_support`. Centers
  are never refitted when models are added; any recalibration is a new version. To map new data:

  ```python
  from analysis.phase2_attitude.ordinalization import assign_categories, strategy_index
  rd, out_of_support = assign_categories('FIP', [0.10, 0.35, 0.90])
  # rd = [1, 3, 5]; out_of_support = [False, False, True]
  # DNC: assign_categories('DNC', [strategy_index(trial['steps']) for trial in trials])
  ```
* **Ordinalization sensitivity.** k-means with k = 4, 5, 6 (`random_state=42`, `n_init=20`) refitted
  on the pooled 600 released raw decisions per task; Kendall's tau between the resulting six-model
  AUC ranks and the frozen-map ranks.

Resampling draws row positions, so exact reproduction of the bootstrap needs the same trial order.
For DNC / Qwen3 Max and DNC / Sonnet 4.5, `reference_row_order.json` gives the order used by the
reference bootstrap. `--file-order` uses the file order instead (point estimates are unchanged;
intervals then differ only by Monte Carlo variation).

## Running

From the repository root:

```bash
python analysis/phase2_attitude/run.py            # full run: 1,000 refits per cell, 100,000 MC draws
python analysis/phase2_attitude/run.py --quick    # smoke run: 40 refits, 20,000 draws -> outputs/phase2_attitude/quick/
```

Options: `--n-boot N`, `--n-mc N`, `--jobs N` (bootstrap worker processes; default: number of CPUs,
at most 18), `--file-order`, `--skip-figures`, and `--human-points CSV` (columns `task, auc, beta`)
to add the pooled human reference as hatched bars in the AUC/slope chart. Without it the chart
shows the six models only.

Runtime: about 5 minutes for the full run with 16 worker processes (the 18,000 bootstrap refits
dominate, and the quasi-separated CTD cells are the slowest); the quick run takes about 30 seconds.
Requires numpy, pandas, scipy, statsmodels, scikit-learn and matplotlib.

Several CTD cells (Qwen3 Max, Sonnet 4.5, occasionally Gemini 3 Pro) are quasi-separated in some
bootstrap resamples: the likelihood keeps increasing as beta goes to minus infinity, so BFGS stops
at a point that depends on the numerical library build. Point estimates, AUC intervals and all
orderings are unaffected at the reported precision, but the extreme lower slope bounds of these
cells can shift between scipy/numpy versions. Such intervals signal quasi-separation, not precise
magnitudes.

## Outputs (`outputs/phase2_attitude/`)

| File | Paper location |
|---|---|
| `main_uncertainty_table.tex` | Table "Risk attitude and sensitivity estimates" (`tab:cr_uncertainty`) |
| `uncertainty_table.tex` | appendix variant of the same table, without Wald p |
| `diagnostics_table.tex` | Table "Observed belief support, response-category counts and support-restricted mean curve height" (`tab:cr_diagnostics`) |
| `mean_decision_rank_table.tex` | Table "AUC rank / raw mean-decision rank" (`tab:cr_mean_ranks`) |
| `pairwise_ordering_table.tex` | Table "Pairwise-probability AUC orderings" (`tab:pairwise_ordering`) |
| `risk_rank_table.tex` | Table "Descriptive point-estimate ranks of six models" (`tab:risk_rank`) |
| `olr_scurves_by_model.png` | Figure "Belief-to-decision mappings across tasks and models" (`fig:validity-curves`) |
| `Histgram_RiskAttitude&RiskSensitivity.png` | Figure "Cross-domain comparison of risk attitude bias and risk sensitivity" (`fig:Histgram_RA&RS`) |
| `model_diagnostics.csv` | per-cell AUC, AUC_S, beta, SE, Wald p, BH p, $B_C$ range, category counts, mean $R_D$ |
| `bootstrap_draws.csv`, `bootstrap_summary.csv` | all refits; median and percentile intervals |
| `kendall_w_draws.csv`, `kendall_w_summary.csv` | W per bootstrap draw; point W, median and 2.5-97.5% range, identical-order count (Section "Cross-Domain Ordering Consistency", Appendix "Uncertainty in Cross-Task Ordering") |
| `pairwise_probabilities.csv`, `pairwise_rankings.csv` | P(AUC_a > AUC_b) per task; task-wise and consensus orders |
| `auc_vs_mean_decision_rank.csv` | Kendall tau between AUC and raw mean-decision ranks |
| `frozen_ordinalization_v1.json`, `frozen_category_mappings.csv` | Appendix "Frozen Ordinalization for Prospective Reuse" (centers) |
| `ordinalization_sensitivity_scores.csv`, `ordinalization_rank_sensitivity.csv` | same appendix (k = 4/6 Kendall correlations) |
| `trial_features.csv` | extracted $B_C$, raw decision, $R_D$ and support flag per frontier trial |
| `summary.json` | counts quoted in the text (significant slopes, intervals containing zero, category use, rank disagreements, W, orders) |
| `earlier_models/earlier_rank_table.tex` | Table "Risk attitude ranks of earlier-generation models" (`tab:rank_earlier`) |
| `earlier_models/AUC_Barchart_AllModels.png` | Figure "Risk attitude bias (AUC) for earlier-generation models" (`fig:auc_earlier`) |
| `earlier_models/Beta_Barchart_AllModels.png` | Figure "OLR slope estimates for earlier-generation models" (`fig:beta_earlier`) |
| `earlier_models/earlier_point_estimates.csv` | earlier-model AUC, beta and diagnostics |

The curve bands in `olr_scurves_by_model.png` are approximate pointwise 95% bands from 300
multivariate-normal coefficient draws (`default_rng(42)`) using each fit's covariance matrix; they
are not trial-bootstrap intervals. The earlier-model AUC chart orders models by mean AUC; the slope
chart orders them by median slope and clips the axis at max(Q1 - 2 IQR, -40), annotating clipped bars.
