"""Regenerate the matched Base -> Instruct ablation tables from the public dataset.

Usage:
    python analysis/ablation/run.py                 # 10,000 bootstrap / permutation resamples
    python analysis/ablation/run.py --n-boot 2000   # quicker, intervals/p-values will differ slightly

Outputs (outputs/ablation/):
    ablation_table.csv / .tex             valid/attempted, task-average P1/P2, task pass
    ablation_dispersion_table.csv / .tex  within-condition B_C SD and normalized R_D entropy
    completion_fisher.csv                 completion yield, Fisher exact tests, Holm
    condition_metrics.csv                 per-scenario n, attempts, mean/SD/RSD, modal share, entropy
    summary.txt                           statements quoted in the paper text
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from riskattitude.paths import output_dir, require_data  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import base_instruct as bi  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n-boot', type=int, default=10_000, help='bootstrap resamples (default 10000)')
    ap.add_argument('--n-perm', type=int, default=10_000, help='permutation resamples (default 10000)')
    ap.add_argument('--seed', type=int, default=bi.RNG_SEED, help='base seed (default 42, as published)')
    args = ap.parse_args()

    require_data()
    start = time.perf_counter()
    out = output_dir('ablation')
    cells, complete = bi.load_cells()
    if not complete:
        print('WARNING: per-run *_meta.json sidecars not found; attempted counts and Fisher tests are '
              'skipped. Re-run data/download_data.py without --skip-meta.')

    rel = bi.reliability_table(cells)
    comp = bi.completion_table(cells)
    disp = bi.dispersion_table(cells, args.n_boot, args.n_perm, args.seed)
    cond = bi.condition_table(cells)

    rel.to_csv(out / 'ablation_table.csv', index=False)
    (out / 'ablation_table.tex').write_text(bi.reliability_latex(rel), encoding='utf-8')
    disp.to_csv(out / 'ablation_dispersion_table.csv', index=False)
    (out / 'ablation_dispersion_table.tex').write_text(bi.dispersion_latex(disp), encoding='utf-8')
    comp.to_csv(out / 'completion_fisher.csv', index=False)
    cond.to_csv(out / 'condition_metrics.csv', index=False)
    lines = bi.summary_lines(rel, comp, disp, args.n_boot, args.n_perm)
    (out / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(rel.to_string(index=False, float_format=lambda x: f'{x:.1f}'))
    print()
    print('\n'.join(lines))
    print(f'\nOutputs in {out} ({time.perf_counter() - start:.1f} s)')


if __name__ == '__main__':
    main()
