"""Participant-cluster bootstrap and matched-exposure subsampling of the task-pooled human OLR.

Each draw resamples whole participants (their retained trials stay together):
  - cluster bootstrap: all participants of the task, with replacement;
  - matched exposure:  20 participants without replacement (about 100 trials, the LLM cell size).
Both use one generator, consumed task by task (CTD, FIP, DNC) and, within a draw, cluster
draw first; keeping this order reproduces the published draws for a given seed.
"""
import numpy as np
import pandas as pd

from fit import fit_olr

SEED = 20260727
MATCHED_PARTICIPANTS = 20
RNG_TASK_ORDER = ('CTD', 'FIP', 'DNC')


def _fit_blocks(blocks, ids):
    bc = np.concatenate([blocks[i][0] for i in ids])
    rd = np.concatenate([blocks[i][1] for i in ids])
    fit = fit_olr(bc, rd)
    return dict(n_participants=len(ids), n_trials=len(bc),
                auc=fit.auc if fit else np.nan, beta=fit.beta if fit else np.nan)


def resample(human, n_boot=1000, seed=SEED, progress=True):
    """Return (cluster_draws, matched_draws), one row per task and draw."""
    rng = np.random.default_rng(seed)
    cluster, matched = [], []
    for task in RNG_TASK_ORDER:
        rows = human[human['task'] == task]
        ids = rows['participant'].unique()
        blocks = {i: (g['bc'].to_numpy(float), g['rd'].to_numpy(int)) for i, g in rows.groupby('participant')}
        for draw in range(1, n_boot + 1):
            sample = rng.choice(ids, size=len(ids), replace=True)
            cluster.append(dict(task=task, draw=draw, **_fit_blocks(blocks, sample)))
            sample = rng.choice(ids, size=min(MATCHED_PARTICIPANTS, len(ids)), replace=False)
            matched.append(dict(task=task, draw=draw, **_fit_blocks(blocks, sample)))
            if progress and draw % 250 == 0:
                print(f'  {task}: {draw}/{n_boot} draws', flush=True)
    return pd.DataFrame(cluster), pd.DataFrame(matched)


def summarize(draws):
    """Median and 2.5th/97.5th percentiles of AUC, beta and trial count over successful draws."""
    out = []
    for task, g in draws.groupby('task', sort=False):
        g = g.dropna(subset=['auc', 'beta'])
        row = dict(task=task, successful_draws=len(g))
        for metric in ('auc', 'beta', 'n_trials'):
            values = g[metric].to_numpy(float)
            row[f'{metric}_median'] = float(np.median(values))
            row[f'{metric}_lo'] = float(np.quantile(values, 0.025))
            row[f'{metric}_hi'] = float(np.quantile(values, 0.975))
        out.append(row)
    return pd.DataFrame(out)
