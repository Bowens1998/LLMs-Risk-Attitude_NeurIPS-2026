"""Phase-1 repeatability metrics (P1, P2) and the reliability gate.

Within each frozen scenario:
    P1 = sample relative standard deviation of B_C, 100 * s / mean (n-1 denominator);
         a constant zero-belief scenario is assigned P1 = 0.
    P2 = share of trials in the modal R_D category (%).
For each model-task cell the three scenario values are averaged with equal weight.
A task passes if mean P1 <= 20% and mean P2 >= 80%; a model passes the overall gate
only if all three tasks pass. Thresholds are applied to unrounded values.
"""
import json
from collections import Counter

import numpy as np

from riskattitude.paths import EARLIER_RELIABILITY, FRONTIER_RELIABILITY, TASKS

P1_MAX = 20.0
P2_MIN = 80.0

FRONTIER_MODELS = {
    'ClaudeSonnet4.5': 'Sonnet 4.5',
    'DeepSeekV3.2': 'DeepSeek V3.2',
    'GPT5.2': 'GPT 5.2',
    'Qwen3Max': 'Qwen3 Max',
    'Gemini3Pro': 'Gemini 3 Pro',
    'Grok4': 'Grok 4',
}
EARLIER_MODELS = {
    'DeepSeekV3': 'DeepSeek V3',
    'Gemini2.5Pro': 'Gemini 2.5 Pro',
    'GPT4.1': 'GPT-4.1',
    'Grok3': 'Grok 3',
    'Qwen3': 'Qwen 3',
    'Sonnet4': 'Sonnet 4',
}
COHORTS = {
    'frontier': (FRONTIER_RELIABILITY / 'Reliability_ALL_merged.json', FRONTIER_MODELS),
    'earlier': (EARLIER_RELIABILITY / 'EarlierModels_Reliability_ALL_merged.json', EARLIER_MODELS),
}


def load_records(cohort):
    path, _ = COHORTS[cohort]
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def scenario_metrics(beliefs, decisions):
    bc = np.asarray(beliefs, float)
    rd = np.asarray(decisions, int)
    mean = float(bc.mean())
    sd = float(bc.std(ddof=1))
    if mean:
        rsd = 100 * sd / mean
    else:
        rsd = 0.0 if sd == 0 else float('inf')
    mode, count = Counter(rd.tolist()).most_common(1)[0]
    return dict(n=len(bc), bc=bc, rd=rd, mean=mean, sd=sd, rsd=rsd, mode=int(mode), modal=100 * count / len(rd))


def task_metrics(records, model, task, p1_max=P1_MAX, p2_min=P2_MIN):
    """Scenario metrics and task-level gate for one model-task cell."""
    cell = [r for r in records if r['model_key'] == model and r['experiment'] == task]
    scenarios = []
    for level in sorted({r['level'] for r in cell}):
        rows = [r for r in cell if r['level'] == level]
        s = scenario_metrics([r['contextual_belief'] for r in rows], [r['risk_decision'] for r in rows])
        s['scenario'] = level
        scenarios.append(s)
    p1 = float(np.mean([s['rsd'] for s in scenarios]))
    p2 = float(np.mean([s['modal'] for s in scenarios]))
    return dict(model=model, task=task, n=len(cell), scenarios=scenarios, p1=p1, p2=p2,
                p1_pass=p1 <= p1_max, p2_pass=p2 >= p2_min, passed=p1 <= p1_max and p2 >= p2_min)


def compute_cohort(cohort, p1_max=P1_MAX, p2_min=P2_MIN):
    """{(task, model): task_metrics} for every model and task of a cohort."""
    records = load_records(cohort)
    models = COHORTS[cohort][1]
    return {(t, m): task_metrics(records, m, t, p1_max, p2_min) for m in models for t in TASKS}


def overall_pass(results, model):
    return all(results[(t, model)]['passed'] for t in TASKS)


def threshold_grid(cohort, p1_grid=(15, 20, 25), p2_grid=(75, 80, 85)):
    """Overall pass/fail of every model under each (P1, P2) threshold pair."""
    models = COHORTS[cohort][1]
    rows = []
    for p1_max in p1_grid:
        for p2_min in p2_grid:
            res = compute_cohort(cohort, p1_max, p2_min)
            row = dict(p1_max=p1_max, p2_min=p2_min)
            for m in models:
                row[m] = overall_pass(res, m)
                row.update({f'{m}/{t}': res[(t, m)]['passed'] for t in TASKS})
            rows.append(row)
    return rows
