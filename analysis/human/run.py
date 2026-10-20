"""Human-reference analyses: B_C / R_D extraction, task-pooled OLR, participant-cluster bootstrap,
matched-exposure subsampling, participant-random-intercept CLMM (R, optional), tables and figures.

Usage (from the repository root):
    python analysis/human/run.py                    # full run; R step if Rscript is on PATH
    python analysis/human/run.py --rscript /path/to/Rscript
    python analysis/human/run.py --skip-r           # Python only
    python analysis/human/run.py --n-boot 100 --no-figures   # quick check
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd  # noqa: E402

from riskattitude.paths import output_dir, require_data  # noqa: E402

from bootstrap import SEED, resample, summarize  # noqa: E402
from extract import load_human, load_llm  # noqa: E402
from fit import fit_olr  # noqa: E402
from tables import primary_table, random_intercept_table  # noqa: E402

HERE = Path(__file__).resolve().parent
ORDER = ['DNC', 'CTD', 'FIP']


def pooled_estimates(human):
    rows = []
    for task in ORDER:
        g = human[human['task'] == task]
        fit = fit_olr(g['bc'], g['rd'])
        rows.append(dict(task=task, n_participants=g['participant'].nunique(), n_trials=len(g),
                         auc=fit.auc, beta=fit.beta))
    return pd.DataFrame(rows)


def run_r(rscript, out, with_ensemble):
    cmd = [rscript, str(HERE / 'clmm.R'), str(out / 'human_long.csv'), str(out / 'human_clmm_results.csv')]
    if with_ensemble:
        cmd += [str(out / 'llm_long.csv'), str(out / 'llm_ensemble_clmm.csv')]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n-boot', type=int, default=1000, help='bootstrap and matched-exposure draws per task')
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--skip-r', action='store_true', help='do not run the R mixed models')
    ap.add_argument('--rscript', help='path to Rscript (default: Rscript found on PATH)')
    ap.add_argument('--no-figures', action='store_true')
    args = ap.parse_args()
    require_data()
    out = output_dir('human')
    start = time.time()

    human, sample = load_human()
    llm = load_llm()
    human.to_csv(out / 'human_long.csv', index=False)
    llm.to_csv(out / 'llm_long.csv', index=False)
    sample = sample.set_index('task').loc[ORDER].reset_index()
    sample.to_csv(out / 'human_sample.csv', index=False)
    print('Retained human trials:\n' + sample.to_string(index=False))

    points = pooled_estimates(human)
    points.to_csv(out / 'human_point_estimates.csv', index=False)
    print('\nTask-pooled OLR:\n' + points.to_string(index=False))

    print(f'\nParticipant resampling ({args.n_boot} draws per task, seed {args.seed}) ...')
    cluster, matched = resample(human, args.n_boot, args.seed)
    cluster.to_csv(out / 'human_cluster_bootstrap_draws.csv', index=False)
    matched.to_csv(out / 'human_matched_draws.csv', index=False)
    cluster_summary, matched_summary = summarize(cluster), summarize(matched)
    cluster_summary.to_csv(out / 'human_cluster_bootstrap_summary.csv', index=False)
    matched_summary.to_csv(out / 'human_matched_summary.csv', index=False)
    print('Participant-cluster bootstrap:\n' + cluster_summary.to_string(index=False))
    print('Matched exposure (20 participants):\n' + matched_summary.to_string(index=False))
    (out / 'human_primary_table.tex').write_text(primary_table(points, cluster_summary), encoding='utf-8')

    rscript = None if args.skip_r else (args.rscript or shutil.which('Rscript'))
    if rscript:
        print('\nCumulative-link mixed models (R ordinal::clmm) ...')
        run_r(rscript, out, with_ensemble=not args.no_figures)
    elif not args.skip_r:
        print('\nRscript not found: skipping the mixed models (pass --rscript or run clmm.R separately).')
    clmm_path = out / 'human_clmm_results.csv'
    if clmm_path.exists():
        clmm = pd.read_csv(clmm_path)
        (out / 'human_random_intercept_table.tex').write_text(random_intercept_table(clmm), encoding='utf-8')

    if not args.no_figures:
        from figures import plot_grouped, plot_pooled_olr
        from groups import load_grouped
        plot_pooled_olr(human, llm, out / 'supplement_human_olr')
        ensemble_path = out / 'llm_ensemble_clmm.csv'
        ensemble = pd.read_csv(ensemble_path) if ensemble_path.exists() else None
        plot_grouped(load_grouped(), out / 'human_groups_vs_llm.png', ensemble, llm)

    print(f'\nDone in {time.time() - start:.0f} s. Outputs: {out}')


if __name__ == '__main__':
    main()
