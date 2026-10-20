"""Phase-1 reliability: metrics, gate, threshold sensitivity, tables and appendix figures.

Usage:
    python analysis/phase1_reliability/run.py               # tables + 36 PDF figures
    python analysis/phase1_reliability/run.py --no-figures  # tables only
    python analysis/phase1_reliability/run.py --png         # also write 300-dpi PNG previews
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from riskattitude.paths import TASKS, output_dir, require_data  # noqa: E402
import metrics  # noqa: E402
import tables  # noqa: E402
from convergence import plot_convergence  # noqa: E402
from figures import plot_consistency  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--no-figures', action='store_true', help='skip the five-panel figures')
    ap.add_argument('--png', action='store_true', help='also save PNG previews of the figures')
    args = ap.parse_args()
    require_data()
    out = output_dir('phase1_reliability')

    results = {c: metrics.compute_cohort(c) for c in metrics.COHORTS}
    frontier, earlier = results['frontier'], results['earlier']
    (out / 'table_ctx_consistency.tex').write_text(tables.table_ctx_consistency(frontier), encoding='utf-8')
    (out / 'table_rd_consistency.tex').write_text(tables.table_rd_consistency(frontier), encoding='utf-8')
    (out / 'conditions_table.tex').write_text(tables.table_scenarios(frontier), encoding='utf-8')
    (out / 'table_earlier_reliability.tex').write_text(tables.table_earlier(earlier), encoding='utf-8')
    tables.write_metrics_csv(out / 'reliability_metrics.csv', results)
    grid = metrics.threshold_grid('frontier')
    tables.write_grid_csv(out / 'threshold_grid.csv', grid)

    for cohort, res in results.items():
        names = metrics.COHORTS[cohort][1]
        print(f'\n{cohort} models (task-average P1 / P2; pass = all three tasks)')
        for m, name in names.items():
            cells = '  '.join(f'{t} {res[(t, m)]["p1"]:5.1f}/{res[(t, m)]["p2"]:5.1f}{"" if res[(t, m)]["passed"] else "*"}'
                              for t in TASKS)
            print(f'  {name:15s} {cells}  -> {"PASS" if metrics.overall_pass(res, m) else "FAIL"}')
    default = next(r for r in grid if r['p1_max'] == metrics.P1_MAX and r['p2_min'] == metrics.P2_MIN)
    same = sum(all(r[m] == default[m] for m in metrics.FRONTIER_MODELS) for r in grid)
    print(f'\nThreshold grid: overall partition unchanged in {same}/{len(grid)} combinations (* = task fails)')

    if not args.no_figures:
        fig_dir = output_dir('phase1_reliability/figures')
        plot_convergence(metrics.load_records('frontier'), fig_dir / 'Convergence.png')
        for cohort, res in results.items():
            prefix = 'reliability_earlier_' if cohort == 'earlier' else 'reliability_'
            for (t, m), r in res.items():
                stem = f'{prefix}{t}_{m}'  # model keys contain dots, so build names explicitly
                paths = [fig_dir / f'{stem}.pdf'] + ([fig_dir / f'{stem}.png'] if args.png else [])
                plot_consistency(r, paths)
        print(f'Figures written to {fig_dir}')
    print(f'Tables written to {out}')


if __name__ == '__main__':
    main()
