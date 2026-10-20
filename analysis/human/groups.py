"""Inputs of the grouped descriptive display (Figure "Human behavioral groups and the LLM reference").

This display uses its own construction, which differs from `extract.py`:
  - source: the all-submissions workbook NeurIPS2026_RiskAttitude_HumanData.xlsx (109 submissions,
    108 unique IDs). Its FIP payloads, and two DNC payloads, are cut at Excel's 32,767-character
    cell limit: FIP trials are recovered by pattern matching on the truncated text and the two
    truncated DNC payloads are skipped;
  - outcomes: CTD ESI (trials with at least one risk-score update); FIP allocation score
    (L + 3M + 5H) / (L + M + H); DNC strategy index on steps with nonzero nudge (at least two),
    scored 5 at the SI floor and 4/3/2/1 by the quartiles of the remaining SI values;
  - groups: k-means (k = 3, standardized per-participant task means) labelled by overall mean.
"""
import json
import re

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from riskattitude.paths import HUMAN_DATA

WORKBOOK = 'NeurIPS2026_RiskAttitude_HumanData.xlsx'
GROUP_NAMES = ('Cautious', 'Neutral', 'Aggressive')
SI_EPS = 0.001
SI_FLOOR = -6.907 + 0.01


def _ctd(payload):
    out = []
    for trial in json.loads(payload).get('trials', []):
        final = trial.get('finalized', {})
        if not isinstance(final, dict) or final.get('ESI') is None or final.get('ctxUpdates', 0) == 0:
            continue
        ctx = [u['ctx'] for s in trial.get('steps', []) for u in s.get('BC_updates', []) if 'ctx' in u]
        if ctx:
            out.append((float(ctx[-1]), float(max(1, min(5, int(final['ESI']))))))
    return out


def _fip(payload):
    allocs = re.findall(r'"alloc"\s*:\s*(\{"[LMH]".*?\})', payload)
    risks = re.findall(r'"risk"\s*:\s*([\d.]+)', payload)
    out = []
    for alloc, risk in zip(allocs, risks):
        a = json.loads(alloc)
        low, mid, high = (float(a.get(k, 0)) for k in ('L', 'M', 'H'))
        if low + mid + high > 0:
            out.append((float(risk), (low + 3 * mid + 5 * high) / (low + mid + high)))
    return out


def _dnc(payload):
    try:
        trials = json.loads(payload).get('trials', [])
    except json.JSONDecodeError:  # truncated cell
        return []
    out = []
    for trial in trials:
        drift = [s for s in trial.get('steps', []) if abs(s.get('nudge', 0)) > 0]
        if trial.get('context_belief') is None or len(drift) < 2:
            continue
        hpr = sum(s.get('action') == 'RIGHT' for s in drift) / len(drift)
        vcr = sum(s.get('action') in ('UP', 'DOWN') for s in drift) / len(drift)
        share = float(np.clip(vcr / (vcr + hpr + SI_EPS), SI_EPS, 1 - SI_EPS))
        out.append((float(trial['context_belief']), float(np.log(share / (1 - share)))))
    return out


def load_grouped(data_dir=HUMAN_DATA):
    """Trial table (task, bc, y, group) for the grouped display; group 0/1/2 = Cautious/Neutral/Aggressive."""
    sheets = pd.read_excel(data_dir / WORKBOOK, sheet_name=['CTD', 'FIP', 'DNC'], dtype={'Subject ID': str})
    rows = []
    for task, parse in (('CTD', _ctd), ('FIP', _fip), ('DNC', _dnc)):
        for subject, payload in zip(sheets[task]['Subject ID'], sheets[task]['Data (JSON)'].astype(str)):
            rows += [dict(subject=subject, task=task, bc=bc, y=y) for bc, y in parse(payload)]
    trials = pd.DataFrame(rows)

    dnc = trials['task'] == 'DNC'
    si = trials.loc[dnc, 'y'].to_numpy()
    q25, q50, q75 = np.percentile(si[si > SI_FLOOR], [25, 50, 75])
    trials.loc[dnc, 'y'] = np.select([si <= SI_FLOOR, si <= q25, si <= q50, si <= q75], [5.0, 4.0, 3.0, 2.0], 1.0)

    # Participants in order of first appearance; neutral midpoint for a task without trials.
    subjects = list(dict.fromkeys(pd.concat([sheets[t]['Subject ID'] for t in ('CTD', 'FIP', 'DNC')])))
    means = trials.groupby(['subject', 'task'])['y'].mean().unstack()
    means = means.reindex(index=subjects, columns=['CTD', 'FIP', 'DNC'])
    means = means[means['CTD'].notna() | means['DNC'].notna()].fillna(3.0)
    features = means.to_numpy()
    raw = KMeans(n_clusters=3, random_state=42, n_init=20).fit_predict(StandardScaler().fit_transform(features))
    rank = np.argsort(np.argsort([features[raw == k].mean() for k in range(3)]))
    group = dict(zip(means.index, rank[raw]))

    trials = trials[trials['subject'].isin(group)].copy()
    trials['group'] = trials['subject'].map(group)
    return trials.drop(columns='subject').reset_index(drop=True)
