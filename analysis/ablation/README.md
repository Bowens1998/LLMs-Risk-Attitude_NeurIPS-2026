# Matched Base → Instruct comparisons

Regenerates the appendix section *Matched Base--Instruct Comparisons* from the public
dataset: Qwen3-30B-A3B Base vs Instruct-2507 and gemma-4-E4B Base vs IT, three frozen
scenarios per task, Phase-1 (reliability) protocol.

```bash
python data/download_data.py            # once; keep the *_meta.json sidecars (no --skip-meta)
python analysis/ablation/run.py         # ~3 s on a laptop CPU
python analysis/ablation/run.py --n-boot 2000 --n-perm 2000   # faster, not the published numbers
```

| Option | Default | Meaning |
|---|---|---|
| `--n-boot` | 10000 | condition-stratified bootstrap resamples (95% percentile intervals) |
| `--n-perm` | 10000 | condition-stratified permutations (two-sided p) |
| `--seed` | 42 | base seed; the published values use the default |

## Inputs

All from `data/hf/FrontierModel_reliability_N90/`:

* `Reliability_{DNC,CTD,FIP}_merged.json` — valid trial records for the model keys
  `Gemma4E4b`, `Gemma4E4bIt`, `Qwen330bA3bBase`, `Qwen330bA3bInstruct2507`.
* The per-run `*_meta.json` sidecars stored next to the per-run files — only for the
  number of **attempted** trials per scenario (`configs[].attempts`), the frozen-scenario
  hashes and decoding settings. Attempted counts are not recoverable from the merged files;
  without the sidecars the script still runs but leaves attempted counts and Fisher tests
  empty. The script checks that each per-run file holds exactly the merged records.

## Outputs (`outputs/ablation/`) and paper locations

| File | Paper location | Content |
|---|---|---|
| `ablation_table.tex` / `.csv` | Table `tab:cr_ablation` ("Matched Base → Instruct results, conditional on valid completion for P1/P2 …") | valid/attempted, task-average P1 (mean scenario RSD, %), P2 (mean modal share, %), task pass (P1 ≤ 20 and P2 ≥ 80) |
| `ablation_dispersion_table.tex` / `.csv` | Table `tab:cr_ablation_dispersion` ("Matched Base--Instruct changes in behavioral dispersion") | within-condition B_C SD and normalized R_D entropy, Δ = Instruct − Base, 95% bootstrap CI, Holm-adjusted permutation p |
| `completion_fisher.csv` | text after the tables ("Completion improves in the four CTD/DNC comparisons (Holm-adjusted Fisher p ≤ 2.52×10⁻⁸) … FIP … p_H = 0.492") | completion yield, two-sided Fisher exact test, Holm over the six comparisons; also checks that Base and Instruct share frozen scenarios and decoding settings |
| `condition_metrics.csv` | supporting detail | per scenario: n, attempted, B_C mean/SD/RSD, modal R_D and share, entropy |
| `summary.txt` | text statements (0/6 Base and 1/6 Instruct task passes; no checkpoint passes all three tasks; 5/6 modal-share increases and RSD decreases; 4/6 SD and 5/6 entropy reductions with p_H = 0.0006) | |

The two `.tex` files reproduce the published tables byte for byte with the default settings.

## Definitions

* **P1 / P2** (per scenario, then averaged over the three scenarios with equal weight):
  sample RSD = 100·SD(n−1)/mean of B_C (a constant zero-belief scenario counts as 0);
  modal share = proportion of trials in the most frequent R_D category. A task passes if
  mean P1 ≤ 20% and mean P2 ≥ 80% (unrounded); a checkpoint passes overall only if all
  three tasks pass.
* **Within-condition B_C SD**: √(Σ squared deviations from each scenario mean / Σ(nₛ − 1)).
* **Normalized R_D entropy**: mean over scenarios of H(R_D)/log 5.
* **Resampling**: bootstrap draws trials with replacement within each scenario, separately
  for Base and Instruct; the permutation test shuffles Base/Instruct labels within each
  scenario; p = (1 + #{|Δ*| ≥ |Δ|}) / (B + 1). Holm correction is applied across the six
  family–task comparisons for each endpoint (and for the Fisher tests).
* **Seeds**: comparison *i* (0–5, ordered Gemma FIP, CTD, DNC, Qwen FIP, CTD, DNC) uses
  `numpy.random.default_rng(seed + offset + 1 + i)` with offset 0 (SD bootstrap),
  100 (SD permutation), 200 (entropy bootstrap), 300 (entropy permutation).

Behavioural statistics are conditional on valid completion: invalid generations count
as completion failures and are never imputed, so differences in completion can change the
population of completed responses.
