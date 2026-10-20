# Benchmarking Risk Attitudes of LLMs

Code for the NeurIPS 2026 paper (Track on Evaluations and Datasets):

> Bowen Sun\*, Rui Min\*, Xianyao Li, Yuxi Wang, Yang Ye, Qi R. Wang, Jing Du.
> **Benchmarking Risk Attitudes of LLMs.** NeurIPS 2026. (\* equal contribution)

The framework separates **contextual belief** $B_C$ (perceived danger, 0–100) from
**risk decision** $R_D$ (an ordinal action, 1 = most cautious, 5 = most aggressive). It applies
this separation across three tasks:

| Task | Setting | $B_C$ | $R_D$ |
|---|---|---|---|
| **DNC** — Drone Navigation Control | navigate a drone through a hazardous grid | post-trial danger rating | Strategy Index of the trajectory, mapped to five frozen categories |
| **CTD** — Clinical Triage Decision | triage an evolving emergency patient | final patient risk score | finalized ESI level |
| **FIP** — Financial Investment Portfolio | allocate across low/medium/high-risk assets | market risk score | high-risk share, mapped to five frozen categories |

This repository contains the task environments, the LLM runner, the human-study interface, and
the analysis code that regenerates every table and figure of the paper.

## Installation

```bash
git clone https://github.com/Bowens1998/LLMs-Risk-Attitude_NeurIPS-2026.git
cd LLMs-Risk-Attitude_NeurIPS-2026
pip install -r requirements.txt          # Python ≥ 3.10 (tested with 3.13)
python data/download_data.py             # dataset from Hugging Face into data/hf/ (~80 MB)
```

The dataset is hosted at
[huggingface.co/datasets/BowenSun1998/Benchmarking_Risk_Attitudes_of_LLMs](https://huggingface.co/datasets/BowenSun1998/Benchmarking_Risk_Attitudes_of_LLMs)
(CC BY 4.0).

On Windows, clone into a short path such as `C:\src`: some dataset file paths are long, and
deep folders can exceed the default 260-character path limit.

R is optional. It is needed only for the participant random-intercept models
(`analysis/human/clmm.R`; R ≥ 4.3 with the `ordinal` package).

## Reproducing the paper

```bash
python analysis/run_pipeline.py                  # all tables and figures (~15 min; R used if on PATH)
python analysis/run_pipeline.py --quick --skip-r # fast smoke run with few resamples, written to outputs/quick/
```

All results are written to `outputs/<module>/`. Each module can also be run on its own:

| Step | Command | Paper |
|---|---|---|
| Phase-1 reliability | `python analysis/phase1_reliability/run.py` | measurement reliability: Tables of belief and decision consistency, the convergence figure, threshold sensitivity, appendix consistency figures (frontier and earlier models) |
| Human reference | `python analysis/human/run.py` | human task-pooled curves, participant-cluster and matched-exposure intervals, random-intercept models, human figures |
| Phase-2 risk attitude | `python analysis/phase2_attitude/run.py` | OLR curves, AUC and slope estimates with bootstrap intervals, cross-task ordering (Kendall's W, pairwise probabilities), frozen ordinalization, earlier-model ranks |
| Base → Instruct | `python analysis/ablation/run.py` | matched Base/Instruct comparisons |

Each module README maps every output file to its table or figure in the paper:
[phase1_reliability](analysis/phase1_reliability/README.md) ·
[human](analysis/human/README.md) ·
[phase2_attitude](analysis/phase2_attitude/README.md) ·
[ablation](analysis/ablation/README.md).

## Definitions

**Phase 1: reliability.** Each model sees three frozen scenarios per task, each repeated up to 30
times. Within a scenario:

- **P1** is the sample relative standard deviation of $B_C$.
- **P2** is the share of the modal $R_D$ category.

Both metrics are averaged over the three scenarios with equal weight.

- A **task passes** if mean P1 ≤ 20% and mean P2 ≥ 80%.
- A **model passes the reliability gate only if all three tasks pass.**

The five models that pass form the cohort for the cross-task ordering analyses.

**Phase 2: risk attitude.** For each model–task cell, an ordered logistic regression of $R_D$ on
$x = B_C/100$ is fitted:

$$\mathrm{logit}\,P(R_D \le k \mid x) = \theta_k - \beta x,\quad k = 1,\dots,4.$$

- **Risk attitude bias** is $\mathrm{AUC} = \int_0^1 \mathbb{E}[R_D \mid x]\,dx \in [1,5]$; lower
  values are more cautious.
- **Risk sensitivity** is $\beta$.
- Uncertainty comes from 1,000 trial-bootstrap refits per cell.
- Cross-task ordering uses Kendall's W and pairwise ordering probabilities from 100,000 Monte
  Carlo draws.

**Ordinalization (benchmark v1).** DNC and FIP raw decisions are assigned to the nearest of five
frozen centers (`experiments/config/frozen_ordinalization_map.json`). The centers are never
refitted when new models are added; any recalibration defines a new benchmark version.

**Human reference.** A task-pooled OLR is fitted per task, with participant-cluster bootstrap
intervals, matched-exposure subsampling and a participant random-intercept model.

## Repository structure

```
├── analysis/
│   ├── run_pipeline.py            run every step
│   ├── phase1_reliability/        P1/P2, gate, threshold grid, consistency and convergence figures
│   ├── phase2_attitude/           OLR, AUC, β, bootstrap, ordering, ordinalization, main figures
│   ├── human/                     human extraction, pooled OLR, bootstrap, CLMM (R), figures
│   └── ablation/                  matched Base → Instruct comparisons
├── experiments/                   data collection (see experiments/README.md)
│   ├── runner/                    Phase-1 LLM runner (OpenRouter / Hugging Face endpoints)
│   ├── web/llm/                   Phase-2 browser task environments for LLMs
│   ├── web/human/                 human-participant portal and task pages
│   ├── scoring/                   B_C / R_D extraction and frozen-category assignment
│   └── config/                    frozen scenarios and benchmark-v1 category centers
├── data/download_data.py          fetches the dataset into data/hf/
├── riskattitude/paths.py          repository-relative paths shared by all scripts
├── outputs/                       generated tables and figures (created on run)
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Running new models

`experiments/README.md` describes how to:

- evaluate a new model with the Phase-1 runner and the Phase-2 task environments;
- score its trials with the frozen benchmark-v1 categories;
- deploy the human-study interface.

API keys are read only from environment variables (`OPENROUTER_API_KEY`, `HF_TOKEN`).
Commercial APIs change over time and their sampling is not seeded, so reruns are not expected
to reproduce earlier responses exactly.

## Numerical reproducibility

- The bootstrap, Monte Carlo and plotting jitter use fixed seeds. With the default settings, the
  tables are regenerated at the precision printed in the paper.
- Several CTD cells are quasi-separated in some bootstrap resamples. The extreme lower slope
  bounds of these cells depend on the scipy/numpy build. Point estimates, AUC intervals and
  orderings do not.
- Figures can differ by a pixel across matplotlib versions.

## License and citation

The code is released under the [MIT License](LICENSE). The dataset is released under CC BY 4.0.

```bibtex
@inproceedings{sun2026benchmarking,
  title     = {Benchmarking Risk Attitudes of {LLMs}},
  author    = {Sun, Bowen and Min, Rui and Li, Xianyao and Wang, Yuxi and Ye, Yang and Wang, Qi R. and Du, Jing},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Track on Evaluations and Datasets},
  year      = {2026}
}
```

Contact: Jing Du (eric.du@essie.ufl.edu)
