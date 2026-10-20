"""Trial bootstrap of AUC and slope: resample the trials of each model-task cell with
replacement, refit the OLR, and summarize percentile intervals.

All resampling indices are drawn up front in the order draw -> task -> model from a
single generator, so results do not depend on how the refits are parallelized.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .olr import auc_beta

BOOT_SEED = 20260724


def resample_indices(sizes: list[int], n_boot: int, seed: int = BOOT_SEED) -> list[np.ndarray]:
    """Per cell, an (n_boot, n) array of row positions."""
    rng = np.random.default_rng(seed)
    out = [np.empty((n_boot, n), dtype=np.int64) for n in sizes]
    for draw in range(n_boot):
        for c, n in enumerate(sizes):
            out[c][draw] = rng.integers(0, n, size=n)
    return out


def _refit_cell(args) -> list[tuple[float, float]]:
    bc, rd, indices = args
    rows = []
    for idx in indices:
        try:
            rows.append(auc_beta(bc[idx], rd[idx]))
        except Exception:
            rows.append((np.nan, np.nan))
    return rows


def trial_bootstrap(cells: dict, n_boot: int, seed: int = BOOT_SEED, jobs: int = 1) -> pd.DataFrame:
    """Long table of bootstrap refits: draw, task, model, auc, beta."""
    keys = list(cells)
    indices = resample_indices([len(cells[k][0]) for k in keys], n_boot, seed)
    work = [(cells[k][0], cells[k][1], idx) for k, idx in zip(keys, indices)]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(_refit_cell, work))
    else:
        results = [_refit_cell(w) for w in work]
    rows = [
        {'draw': draw + 1, 'task': task, 'model': model, 'auc': auc, 'beta': beta}
        for (task, model), cell in zip(keys, results)
        for draw, (auc, beta) in enumerate(cell)
    ]
    order = {k: i for i, k in enumerate(keys)}
    draws = pd.DataFrame(rows)
    draws['_cell'] = [order[t, m] for t, m in zip(draws.task, draws.model)]
    return draws.sort_values(['draw', '_cell']).drop(columns='_cell').reset_index(drop=True)


def summarize(draws: pd.DataFrame) -> pd.DataFrame:
    """Median and percentile 95% interval per task, model and metric."""
    rows = []
    for (task, model), g in draws.groupby(['task', 'model']):
        for metric in ('auc', 'beta'):
            v = g[metric].dropna().to_numpy(float)
            rows.append({'task': task, 'model': model, 'metric': metric, 'n_success': len(v),
                         'median': float(np.median(v)),
                         'q025': float(np.quantile(v, 0.025)),
                         'q975': float(np.quantile(v, 0.975))})
    return pd.DataFrame(rows)
