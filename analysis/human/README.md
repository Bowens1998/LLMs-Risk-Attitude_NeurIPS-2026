# Human reference

Task-pooled human belief-to-decision curves, used as a descriptive reference for the LLM
results: extraction of $B_C$ / $R_D$ from the public raw task records, one ordered logistic
regression (OLR) per task, participant-cluster bootstrap intervals, matched-exposure
subsampling, participant-random-intercept cumulative-link mixed models, and the human
tables and figures.

## Definitions

**Cohort.** The first 100 eligible unique participants (`Human_Cohort_First100Unique.csv`),
five trials per task before filtering. A trial is retained when both $B_C$ and $R_D$ can be
extracted. Retained: DNC 484 trials / 98 participants, CTD 500 / 100, FIP 500 / 100.

**Extraction** (`extract.py`; identical rules for humans and LLMs):

| Task | $B_C$ (0–100) | $R_D$ (1 = most cautious) |
|---|---|---|
| DNC | post-trial danger rating `context_belief` | Strategy Index on drift-active steps (`drift_row != 0`; all steps if none), SI = logit(VCR / (VCR + HPR + 0.001)) clipped to [0.001, 0.999], assigned to the nearest frozen center |
| CTD | last Patient Risk Score (`BC_updates[-1].ctx` of the last step that has one) | finalized ESI |
| FIP | Market Risk Score `report.contextual.risk` | high-risk share H / (L + M + H), assigned to the nearest frozen center |

Frozen benchmark-v1 centers (R_D = 1..5): DNC SI (6.868044, 1.078387, 0.113035, −0.834347,
−6.906755); FIP share (0.082306, 0.177340, 0.291722, 0.446522, 0.594595). They are never
refitted on human data.

**OLR** (`fit.py`): logit P(Y ≤ k | x) = θ_k − βx with x = $B_C$/100 (statsmodels
`OrderedModel`, BFGS). AUC = ∫₀¹ E[Y | x] dx, trapezoidal rule on a 201-point grid.

**Uncertainty** (`bootstrap.py`; 1,000 draws, seed 20260727):
- participant-cluster bootstrap: resample all participants of a task with replacement, keeping
  each participant's trials together; percentile 95% intervals;
- matched exposure: 20 participants without replacement per draw (≈100 trials, the size of an
  LLM cell); median and 2.5th–97.5th percentiles. These are subsampling ranges, not confidence
  intervals for between-person dispersion.

**Mixed model** (`clmm.R`): per task, `rd ~ z(bc) + (1 | participant)`, logit link,
`nAGQ = 10`, $B_C$ standardized within the task; Wald 95% intervals.

**Grouped descriptive display** (`groups.py`, `figures.plot_grouped`): all 109 submissions
(108 unique IDs) in `NeurIPS2026_RiskAttitude_HumanData.xlsx`; CTD ESI, FIP allocation score
(L + 3M + 5H)/(L + M + H), DNC percentile-scored SI; k-means (k = 3) groups; LOESS with a
300-draw bootstrap band and binned means ± SEM. The dashed LLM ensemble reference is
`rd ~ x + (1 | model)` fitted to the Phase-2 records of the six frontier models (R).

## Run

From the repository root, after `python data/download_data.py`:

```bash
python analysis/human/run.py                    # everything; R step if Rscript is on PATH
python analysis/human/run.py --rscript /path/to/Rscript
python analysis/human/run.py --skip-r           # Python only (no mixed models, no dashed LLM reference)
python analysis/human/run.py --n-boot 100 --no-figures   # quick check
```

The R step can also be run separately after a Python run, then `run.py --skip-r` builds the
mixed-model table from the existing CSV:

```bash
Rscript analysis/human/clmm.R outputs/human/human_long.csv outputs/human/human_clmm_results.csv \
        outputs/human/llm_long.csv outputs/human/llm_ensemble_clmm.csv
```

Runtime: about 7 minutes in total (6,000 bootstrap/subsampling OLR fits dominate; the R models
and figures take well under a minute).

## Requirements

- Python ≥ 3.10 with numpy, pandas, scipy, statsmodels, scikit-learn, matplotlib, openpyxl
  (the last for the grouped display only).
- Optional: R ≥ 4.3 with the `ordinal` package (`install.packages("ordinal")`; tested with
  R 4.5.2, ordinal 2025.12.29). A non-default library can be supplied through the standard
  `R_LIBS_USER` / `R_LIBS` environment variables.

## Outputs (`outputs/human/`)

| File | Paper |
|---|---|
| `human_primary_table.tex` | Table "Primary-cohort task-pooled human AUC" (`tab:human_primary_summary`); AUCs quoted in Section "LLMs vs. Human Risk Baselines" |
| `human_random_intercept_table.tex`, `human_clmm_results.csv` | Table "Human belief--decision slopes from participant-random-intercept cumulative-logit models" (`tab:cr_human_clmm`), Appendix "Human Fitting Unit and Exposure" |
| `human_sample.csv` | Retained trials and participants (Sections "Participants and Models" and "Human Fitting Unit and Exposure") |
| `human_point_estimates.csv` | Task-pooled OLR AUC and β |
| `human_cluster_bootstrap_summary.csv`, `human_cluster_bootstrap_draws.csv` | Participant-cluster intervals (main text and `tab:human_primary_summary`) |
| `human_matched_summary.csv`, `human_matched_draws.csv` | Matched-exposure medians and ranges, Appendix "Human Fitting Unit and Exposure" |
| `supplement_human_olr.pdf` / `.png` | Figure "Primary task-pooled human OLR curves" (`fig:cr_human_olr`) |
| `human_groups_vs_llm.png` | Figure "Human behavioral groups and the LLM reference" (`fig:HumanVsLLMsBaseline`) |
| `human_long.csv`, `llm_long.csv` | Analysis-ready trials passed to `clmm.R` (participants identified only by public cohort order) |
| `llm_ensemble_clmm.csv` | Parameters of the dashed LLM reference |

## Reproduction notes

- With the defaults (seed 20260727, 1,000 draws) every bootstrap and matched-exposure draw
  reproduces the published values (|ΔAUC| < 1e-8). The draws for FIP and DNC depend on
  `--n-boot`, because one random generator is consumed task by task.
- Mixed-model estimates agree with the published values to about 1e-5 (optimizer tolerance;
  the participant factor here is the cohort index).
- The dashed LLM reference in the grouped display is fitted to the public Phase-2 records.
