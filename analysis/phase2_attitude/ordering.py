"""Cross-task ordering: point ranks, Kendall's W, and pairwise ordering probabilities."""
from __future__ import annotations

import numpy as np
import pandas as pd

MC_SEED = 2741


def ranks_lowest_first(scores: dict[str, float], models) -> dict[str, int]:
    """Rank 1 = lowest score; exact ties keep the given model order."""
    ordered = sorted(models, key=lambda m: scores[m])
    return {m: i + 1 for i, m in enumerate(ordered)}


def kendalls_w(rank_matrix) -> float:
    """Kendall's coefficient of concordance (rows = tasks, columns = models), no tie correction."""
    r = np.asarray(rank_matrix, dtype=float)
    m, n = r.shape
    sums = r.sum(axis=0)
    return float(12 * np.sum((sums - sums.mean()) ** 2) / (m ** 2 * (n ** 3 - n)))


def rank_matrix(scores_by_task: dict, models, tasks) -> list[list[int]]:
    out = []
    for task in tasks:
        r = ranks_lowest_first(scores_by_task[task], models)
        out.append([r[m] for m in models])
    return out


def bootstrap_w(draws: pd.DataFrame, point_scores: dict, models, tasks) -> pd.DataFrame:
    """Kendall's W and order agreement for every joint bootstrap draw."""
    point = rank_matrix(point_scores, models, tasks)
    rows = []
    for draw, g in draws[draws.model.isin(models)].groupby('draw'):
        scores = {t: dict(zip(gt.model, gt.auc)) for t, gt in g.groupby('task')}
        if any(len(scores.get(t, {})) != len(models) or
               not np.all(np.isfinite(list(scores[t].values()))) for t in tasks):
            continue
        matrix = rank_matrix(scores, models, tasks)
        orders = {tuple(sorted(models, key=lambda m: scores[t][m])) for t in tasks}
        rows.append({'draw': draw, 'kendall_W': kendalls_w(matrix),
                     'same_order_all_tasks': len(orders) == 1,
                     'matches_point_ranks': matrix == point})
    return pd.DataFrame(rows)


def monte_carlo_pairwise(draws: pd.DataFrame, models_drawn, models, tasks,
                         n_mc: int = 100_000, seed: int = MC_SEED) -> pd.DataFrame:
    """P(AUC_a > AUC_b) from independent draws of each cell's bootstrap AUC distribution.

    One value is drawn per model and iteration from its empirical bootstrap
    distribution (task-major, model order `models_drawn`); exact ties count one half.
    """
    rng = np.random.default_rng(seed)
    samples = {}
    for task in tasks:
        for model in models_drawn:
            dist = draws[(draws.task == task) & (draws.model == model)].sort_values('draw')
            dist = dist.auc.to_numpy(float)
            dist = dist[np.isfinite(dist)]
            samples[task, model] = dist[rng.integers(0, dist.size, size=n_mc)]
    rows = []
    for task in tasks:
        for a in models:
            for b in models:
                if a == b:
                    continue
                sa, sb = samples[task, a], samples[task, b]
                p = float(np.mean(sa > sb) + 0.5 * np.mean(sa == sb))
                rows.append({'task': task, 'model_a': a, 'model_b': b, 'p_auc_a_greater_b': p,
                             'mcse': float(np.sqrt(p * (1 - p) / n_mc))})
    return pd.DataFrame(rows)


def pairwise_order(p_greater: dict[tuple[str, str], float], models) -> list[dict]:
    """Order models from higher to lower AUC: number of pairwise probabilities above
    0.5 (majority wins), with the sum of probabilities breaking equal-win ties."""
    rows = []
    for a in models:
        probs = [p_greater[a, b] for b in models if b != a]
        rows.append({'model': a, 'majority_wins': sum(p > 0.5 for p in probs),
                     'expected_wins': float(sum(probs))})
    rows.sort(key=lambda r: (-r['majority_wins'], -r['expected_wins'], r['model']))
    for i, r in enumerate(rows, start=1):
        r['rank_high_to_low'] = i
    return rows


def pairwise_rankings(pairs: pd.DataFrame, models, tasks) -> tuple[pd.DataFrame, list[dict]]:
    """Task-wise orders and the equal-task-weight consensus order."""
    table = {(r.task, r.model_a, r.model_b): r.p_auc_a_greater_b for r in pairs.itertuples()}
    per_task = []
    for task in tasks:
        for row in pairwise_order({(a, b): table[task, a, b] for a in models for b in models if a != b}, models):
            per_task.append({'task': task, **row})
    mean_p = {(a, b): float(np.mean([table[t, a, b] for t in tasks]))
              for a in models for b in models if a != b}
    return pd.DataFrame(per_task), pairwise_order(mean_p, models)
