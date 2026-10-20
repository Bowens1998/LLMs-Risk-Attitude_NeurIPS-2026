"""Reproduce every table and figure of the paper from the public dataset.

Usage:
    python data/download_data.py          # once, ~80 MB
    python analysis/run_pipeline.py       # full run (~12 min on a laptop; R optional)
    python analysis/run_pipeline.py --quick --skip-r   # fast smoke run (~1 min)

Steps (outputs/<module>/):
    1. phase1_reliability  repeatability metrics, reliability gate, threshold grid, consistency figures
    2. human               human baseline: extraction, pooled OLR, cluster bootstrap, CLMM (R)
    3. phase2_attitude     OLR, AUC, beta, bootstrap, cross-task ordering, ordinalization, main figures
    4. ablation            matched Base -> Instruct comparisons
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from riskattitude.paths import OUTPUT_DIR, require_data  # noqa: E402


def run(step, script, *args, out_dir=OUTPUT_DIR):
    print(f'\n=== {step} ===', flush=True)
    start = time.time()
    env = dict(os.environ, RISKATTITUDE_OUTPUT_DIR=str(out_dir))
    subprocess.run([sys.executable, str(ROOT / 'analysis' / script / 'run.py'), *map(str, args)],
                   cwd=ROOT, env=env, check=True)
    print(f'--- {step}: {time.time() - start:.0f} s', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quick', action='store_true', help='few resamples, for checking that everything runs')
    ap.add_argument('--skip-r', action='store_true', help='skip the R mixed models (also skipped if Rscript is missing)')
    ap.add_argument('--jobs', type=int, default=None, help='worker processes for the Phase-2 bootstrap')
    args = ap.parse_args()
    require_data()

    skip_r = args.skip_r or shutil.which('Rscript') is None
    if skip_r and not args.skip_r:
        print('Rscript not found on PATH: the R mixed models are skipped (use analysis/human/run.py --rscript PATH).')
    out = OUTPUT_DIR / 'quick' if args.quick else OUTPUT_DIR   # quick runs never overwrite the full results

    run('1. Phase-1 reliability', 'phase1_reliability', out_dir=out)
    human = (['--n-boot', 50] if args.quick else []) + (['--skip-r'] if skip_r else [])
    run('2. Human baseline', 'human', *human, out_dir=out)
    phase2 = ['--human-points', out / 'human' / 'human_point_estimates.csv']
    phase2 += ['--n-boot', 40, '--n-mc', 20_000] if args.quick else []
    phase2 += ['--jobs', args.jobs] if args.jobs else []
    run('3. Phase-2 risk attitude', 'phase2_attitude', *phase2, out_dir=out)
    run('4. Base -> Instruct ablation', 'ablation', *(['--n-boot', 500, '--n-perm', 500] if args.quick else []), out_dir=out)
    print(f'\nAll outputs are in {out}')


if __name__ == '__main__':
    main()
