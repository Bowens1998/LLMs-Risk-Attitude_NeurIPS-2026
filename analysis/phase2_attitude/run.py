#!/usr/bin/env python3
"""Regenerate the Phase-2 risk-attitude results for the frontier and earlier models.

    python analysis/phase2_attitude/run.py              # full run (1,000 bootstrap refits per cell)
    python analysis/phase2_attitude/run.py --quick      # fast smoke run into outputs/phase2_attitude/quick/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

from analysis.phase2_attitude import figures, tables  # noqa: E402
from analysis.phase2_attitude.bootstrap import summarize, trial_bootstrap  # noqa: E402
from analysis.phase2_attitude.olr import fit_olr  # noqa: E402
from analysis.phase2_attitude.ordering import (  # noqa: E402
    bootstrap_w, kendalls_w, monte_carlo_pairwise, pairwise_rankings, rank_matrix,
    ranks_lowest_first)
from analysis.phase2_attitude.ordinalization import (  # noqa: E402
    FROZEN_CENTERS, kmeans_categories, mapping_spec)
from analysis.phase2_attitude.trials import (  # noqa: E402
    EARLIER_LABELS, EARLIER_MODELS, FRONTIER_LABELS, FRONTIER_MODELS, RELIABLE_MODELS, cells,
    load_earlier, load_frontier)
from riskattitude.paths import TASKS, output_dir, require_data  # noqa: E402

TABLE_ORDER = sorted(FRONTIER_MODELS)  # row order of the per-cell tables


def point_estimates(df: pd.DataFrame, models, labels) -> tuple[dict, pd.DataFrame]:
    """OLR fit per task and model plus descriptive diagnostics."""
    fits, rows = {}, []
    for task in TASKS:
        for model in models:
            g = df[(df.task == task) & (df.model == model)]
            fit = fit_olr(g.bc, g.rd)
            fits[task, model] = fit
            counts = g.rd.value_counts()
            rows.append({
                'task': task, 'model': model, 'model_label': labels[model], 'n': len(g),
                'bc_min': g.bc.min(), 'bc_max': g.bc.max(), 'categories': g.rd.nunique(),
                'dominant_percent': 100 * counts.max() / len(g), 'mean_rd': g.rd.mean(),
                'auc': fit.auc, 'auc_support': fit.auc_support, 'beta': fit.beta,
                'beta_se': fit.beta_se, 'beta_p': fit.beta_p,
                **{f'rd{k}': int(counts.get(k, 0)) for k in range(1, 6)},
            })
    points = pd.DataFrame(rows)
    points['beta_p_bh'] = multipletests(points.beta_p, method='fdr_bh')[1]
    return fits, points


def ordinalization_sensitivity(released: pd.DataFrame, points: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Refit k-means (k = 4, 5, 6) on the pooled released action indices of each task and
    compare the six-model AUC ranks with those under the frozen k = 5 map."""
    score_rows, rank_rows = [], []
    for task in ('DNC', 'FIP'):
        data = released[released.task == task].copy()
        primary = ranks_lowest_first(dict(zip(points[points.task == task].model,
                                              points[points.task == task].auc)), FRONTIER_MODELS)
        for k in (4, 5, 6):
            data['alt_rd'] = kmeans_categories(data.raw, task, k)
            scores = {}
            for model in FRONTIER_MODELS:
                g = data[data.model == model]
                fit = fit_olr(g.bc, g.alt_rd)
                scores[model] = fit.auc
                score_rows.append({'task': task, 'method': f'refit_k{k}', 'k': k, 'model': model,
                                   'auc': fit.auc, 'auc_rescaled_1_5': 1 + 4 * (fit.auc - 1) / (k - 1),
                                   'beta': fit.beta})
            alt = ranks_lowest_first(scores, FRONTIER_MODELS)
            tau = kendalltau([primary[m] for m in FRONTIER_MODELS], [alt[m] for m in FRONTIER_MODELS])[0]
            rank_rows.append({'task': task, 'method': f'refit_k{k}', 'k': k, 'kendall_tau_vs_frozen_k5': tau,
                              'order_low_to_high_auc': ' < '.join(sorted(FRONTIER_MODELS, key=scores.get))})
    pd.DataFrame(score_rows).to_csv(out / 'ordinalization_sensitivity_scores.csv', index=False)
    ranks = pd.DataFrame(rank_rows)
    ranks.to_csv(out / 'ordinalization_rank_sensitivity.csv', index=False)
    return ranks


def write_frozen_mapping(out: Path) -> None:
    (out / 'frozen_ordinalization_v1.json').write_text(json.dumps(mapping_spec(), indent=2), encoding='utf-8')
    pd.DataFrame([{'task': t, 'rd': i + 1, 'center': c}
                  for t in ('DNC', 'FIP') for i, c in enumerate(FROZEN_CENTERS[t])]
                 ).to_csv(out / 'frozen_category_mappings.csv', index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-boot', type=int, default=1000, help='bootstrap refits per cell (default 1000)')
    parser.add_argument('--n-mc', type=int, default=100_000, help='Monte Carlo draws for pairwise ordering')
    parser.add_argument('--jobs', type=int, default=min(os.cpu_count() or 1, 18),
                        help='worker processes for the bootstrap refits')
    parser.add_argument('--quick', action='store_true', help='40 refits, 20,000 draws, output to quick/')
    parser.add_argument('--file-order', action='store_true',
                        help='resample trials in file order instead of the reference order')
    parser.add_argument('--human-points', type=Path,
                        help='optional CSV (task, auc, beta) drawn as the human reference in the bar chart')
    parser.add_argument('--skip-figures', action='store_true')
    args = parser.parse_args()
    if args.quick:
        args.n_boot, args.n_mc = 40, 20_000

    require_data()
    out = output_dir('phase2_attitude')
    if args.quick:
        out = out / 'quick'
        out.mkdir(exist_ok=True)
    started = time.time()

    released = load_frontier(reference_order=False)
    frontier = released if args.file_order else load_frontier(reference_order=True)
    released.to_csv(out / 'trial_features.csv', index=False)

    # Point estimates, Wald tests, Benjamini-Hochberg, support diagnostics.
    fits, points = point_estimates(frontier, TABLE_ORDER, FRONTIER_LABELS)
    points.to_csv(out / 'model_diagnostics.csv', index=False)
    tables.diagnostics_table(points, out)
    rank_df = tables.point_rank_tables(points, out)
    print(f'[{time.time() - started:6.1f}s] point estimates')

    # Trial bootstrap.
    draws = trial_bootstrap(cells(frontier, FRONTIER_MODELS), args.n_boot, jobs=args.jobs)
    draws.to_csv(out / 'bootstrap_draws.csv', index=False)
    summary = summarize(draws)
    summary.insert(2, 'model_label', summary.model.map(FRONTIER_LABELS))
    summary.to_csv(out / 'bootstrap_summary.csv', index=False)
    tables.uncertainty_tables(points, summary, args.n_boot, out)
    print(f'[{time.time() - started:6.1f}s] bootstrap ({args.n_boot} refits x {len(TASKS) * len(FRONTIER_MODELS)} cells)')

    # Kendall's W among the five models passing the reliability gate.
    point_scores = {t: dict(zip(points[points.task == t].model, points[points.task == t].auc)) for t in TASKS}
    w_draws = bootstrap_w(draws, point_scores, RELIABLE_MODELS, TASKS)
    w_draws.to_csv(out / 'kendall_w_draws.csv', index=False)
    w_summary = {
        'point_W': kendalls_w(rank_matrix(point_scores, RELIABLE_MODELS, TASKS)),
        'n_draws': len(w_draws),
        'bootstrap_W_median': float(w_draws.kendall_W.median()),
        'bootstrap_W_q025': float(w_draws.kendall_W.quantile(0.025)),
        'bootstrap_W_q975': float(w_draws.kendall_W.quantile(0.975)),
        'n_same_order_all_tasks': int(w_draws.same_order_all_tasks.sum()),
        'n_matching_point_ranks': int(w_draws.matches_point_ranks.sum()),
    }
    pd.DataFrame([w_summary]).to_csv(out / 'kendall_w_summary.csv', index=False)

    # Pairwise ordering probabilities.
    pairs = monte_carlo_pairwise(draws, FRONTIER_MODELS, RELIABLE_MODELS, TASKS, n_mc=args.n_mc)
    pairs.to_csv(out / 'pairwise_probabilities.csv', index=False)
    per_task, consensus = pairwise_rankings(pairs, RELIABLE_MODELS, TASKS)
    ranking = pd.concat([per_task, pd.DataFrame(consensus).assign(task='consensus')], ignore_index=True)
    ranking.to_csv(out / 'pairwise_rankings.csv', index=False)
    task_orders = {t: per_task[per_task.task == t].sort_values('rank_high_to_low').model.tolist()
                   for t in ('DNC', 'FIP', 'CTD')}
    tables.pairwise_ordering_table(task_orders, out)
    print(f'[{time.time() - started:6.1f}s] Kendall W and pairwise ordering ({args.n_mc:,} draws)')

    # AUC versus raw mean-decision ranks.
    tau_rows = []
    for task in TASKS:
        g = points[points.task == task]
        auc_r = ranks_lowest_first(dict(zip(g.model, g.auc)), FRONTIER_MODELS)
        mean_r = ranks_lowest_first(dict(zip(g.model, g.mean_rd)), FRONTIER_MODELS)
        tau_rows.append({'task': task, 'kendall_tau': kendalltau([auc_r[m] for m in FRONTIER_MODELS],
                                                                 [mean_r[m] for m in FRONTIER_MODELS])[0]})
    pd.DataFrame(tau_rows).to_csv(out / 'auc_vs_mean_decision_rank.csv', index=False)

    # Frozen ordinalization and k = 4/5/6 sensitivity.
    write_frozen_mapping(out)
    sensitivity = ordinalization_sensitivity(released, points, out)

    # Earlier-generation models.
    earlier_out = out / 'earlier_models'
    earlier_out.mkdir(exist_ok=True)
    earlier = load_earlier()
    _, earlier_points = point_estimates(earlier, EARLIER_MODELS, EARLIER_LABELS)
    earlier_points.to_csv(earlier_out / 'earlier_point_estimates.csv', index=False)
    earlier_ranks = tables.earlier_rank_table(earlier_points, earlier_out)
    print(f'[{time.time() - started:6.1f}s] ordinalization sensitivity and earlier models')

    if not args.skip_figures:
        human = pd.read_csv(args.human_points) if args.human_points else None
        figures.olr_curves_by_model(fits, out)
        figures.attitude_sensitivity_bars(points, out, human=human)
        figures.observed_support(fits, frontier, out)
        figures.earlier_bars(earlier_points, earlier_out)
        print(f'[{time.time() - started:6.1f}s] figures')

    boot_beta = summary[summary.metric == 'beta']
    zero_in_ci = boot_beta[(boot_beta.q025 <= 0) & (boot_beta.q975 >= 0)]
    report = {
        'n_boot': args.n_boot, 'n_mc': args.n_mc, 'row_order': 'file' if args.file_order else 'reference',
        'negative_slopes': int((points.beta < 0).sum()),
        'wald_significant_by_task': points[points.beta_p < 0.05].task.value_counts().to_dict(),
        'bh_significant': int((points.beta_p_bh < 0.05).sum()),
        'bootstrap_slope_ci_contains_zero': [f'{r.task}/{r.model}' for r in zero_in_ci.itertuples()],
        'cells_using_five_categories': int((points.categories == 5).sum()),
        'dominant_share_range_percent': [points.dominant_percent.min(), points.dominant_percent.max()],
        'auc_vs_mean_rank_differences': int((rank_df.auc_rank != rank_df.mean_rd_rank).sum()),
        'auc_vs_mean_rank_tau': {r['task']: r['kendall_tau'] for r in tau_rows},
        'kendall_w': w_summary,
        'pairwise_orders_high_to_low': task_orders,
        'pairwise_consensus_high_to_low': [r['model'] for r in consensus],
        'ordinalization_tau_vs_frozen': {f'{r.task}_k{r.k}': r.kendall_tau_vs_frozen_k5
                                         for r in sensitivity.itertuples()},
        'earlier_ranks': earlier_ranks.to_dict(orient='records'),
        'runtime_seconds': round(time.time() - started, 1),
    }
    (out / 'summary.json').write_text(json.dumps(report, indent=2, default=float), encoding='utf-8')
    print(f'Done in {time.time() - started:.1f}s. Outputs: {out}')


if __name__ == '__main__':
    main()
