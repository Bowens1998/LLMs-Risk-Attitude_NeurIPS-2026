"""Ordinal risk-decision categories (benchmark v1) and k-means sensitivity refits.

CTD uses the native ESI level (1-5). DNC and FIP produce a continuous action index
that is assigned to the nearest of five frozen centers:

* DNC  Strategy Index  SI = logit(clip(VCR / (VCR + HPR + 0.001), 0.001, 0.999)),
       computed on drift-active steps (all steps if none); higher SI = more cautious.
* FIP  high-volatility share H / (L + M + H); higher share = more aggressive.

The centers are fixed. New observations are mapped without refitting; values outside
the v1 reference range map to the nearest extreme category and are flagged.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.cluster import KMeans

VERSION = '1.0'

# Centers listed in R_D = 1..5 order (1 = most cautious).
FROZEN_CENTERS = {
    'DNC': np.array([6.868043918281026, 1.0783865284913943, 0.11303487689571717,
                     -0.8343466753906355, -6.906754778648555]),
    'FIP': np.array([0.08230593607305913, 0.17734042553191486, 0.2917218543046358,
                     0.44652173913043455, 0.5945945945945944]),
}
REFERENCE_RANGE = {
    'DNC': (-6.906754778648555, 6.906754778648555),
    'FIP': (0.0, 0.8),
}
SOURCE_VARIABLE = {'DNC': 'Strategy Index', 'FIP': 'H/(L+M+H)', 'CTD': 'ESI'}

SI_EPS = 1e-3


def strategy_index(steps: list[dict]) -> float | None:
    """DNC Strategy Index from the step log of one trial."""
    drift = [s for s in steps if s.get('drift_row', 0) != 0]
    work = drift or steps
    if not work:
        return None
    n = len(work)
    vcr = sum(s.get('action') in ('UP', 'DOWN') for s in work) / n
    hpr = sum(s.get('action') == 'RIGHT' for s in work) / n
    share = vcr / (vcr + hpr + SI_EPS)
    share = min(max(share, SI_EPS), 1 - SI_EPS)
    return math.log(share / (1 - share))


def assign_categories(task: str, values) -> tuple[np.ndarray, np.ndarray]:
    """Map raw action indices to R_D with the frozen v1 centers.

    Returns (labels, out_of_support). An exact tie goes to the smaller label.
    """
    values = np.asarray(values, dtype=float)
    if task == 'CTD':
        labels = values.astype(int)
        return labels, ~np.isin(labels, [1, 2, 3, 4, 5])
    centers = FROZEN_CENTERS[task]
    labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1) + 1
    lo, hi = REFERENCE_RANGE[task]
    return labels.astype(int), (values < lo) | (values > hi)


def boundaries(task: str) -> np.ndarray:
    """Midpoints between adjacent centers, ascending in raw-index units."""
    c = np.sort(FROZEN_CENTERS[task])
    return (c[:-1] + c[1:]) / 2


def kmeans_categories(values, task: str, k: int, seed: int = 42) -> np.ndarray:
    """Refit 1-D k-means with k clusters on `values` and return ordered labels 1..k."""
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    km = KMeans(n_clusters=k, random_state=seed, n_init=20).fit(values)
    order = np.argsort(km.cluster_centers_.ravel())
    if task == 'DNC':
        order = order[::-1]
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(1, k + 1)
    return remap[km.predict(values)]


def mapping_spec() -> dict:
    """Machine-readable description of the frozen mapping."""
    spec = {
        'benchmark_mapping_version': VERSION,
        'rule': 'nearest fixed center; exact ties go to the smaller R_D label; values outside '
                'the reference range map to the nearest extreme center and are flagged '
                'out_of_support; any recalibration defines a new version',
        'tasks': {'CTD': {'mapping_type': 'native_ordinal', 'source_variable': 'ESI',
                          'risk_decision_labels': [1, 2, 3, 4, 5]}},
    }
    for task in ('DNC', 'FIP'):
        lo, hi = REFERENCE_RANGE[task]
        spec['tasks'][task] = {
            'mapping_type': 'nearest_fixed_center',
            'source_variable': SOURCE_VARIABLE[task],
            'centers_in_rd_order': FROZEN_CENTERS[task].tolist(),
            'numeric_boundaries_ascending': boundaries(task).tolist(),
            'reference_range': {'minimum': lo, 'maximum': hi},
        }
    spec['tasks']['DNC']['direction'] = 'higher Strategy Index maps to a lower, more cautious R_D'
    spec['tasks']['FIP']['direction'] = 'higher high-volatility share maps to a higher, more aggressive R_D'
    spec['tasks']['DNC']['definition'] = ('SI = logit(clip(VCR/(VCR+HPR+0.001), 0.001, 0.999)) on '
                                          'steps with nonzero drift (all steps if none)')
    return spec
