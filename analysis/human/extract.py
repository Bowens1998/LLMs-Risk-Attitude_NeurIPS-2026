"""Trial-level B_C / R_D extraction for the human cohort and the frontier-model Phase-2 records.

Human inputs are the public raw task records (one row per participant and task, five trials
in the `Data (JSON)` payload). The same extraction rules are applied to the LLM records, and
DNC/FIP decisions are assigned to the frozen benchmark-v1 category centers (never refitted).
"""
import json
import math
import re

import numpy as np
import pandas as pd

from riskattitude.paths import FRONTIER_MAIN, HUMAN_DATA

# HF file names keep the source-tab labels: TPB = CTD, DSB = DNC.
HUMAN_FILES = {'CTD': 'Human_TPB_N100_full.csv', 'FIP': 'Human_FIP_N100_full.csv', 'DNC': 'Human_DSB_N100_full.csv'}
COHORT_FILE = 'Human_Cohort_First100Unique.csv'
MODELS = ('DeepSeekV3.2', 'GPT5.2', 'Gemini3Pro', 'Grok4', 'Qwen3Max', 'Sonnet4.5')

# Frozen benchmark-v1 centers in R_D = 1..5 order (1 = most cautious).
DNC_CENTERS = np.array([6.868043918281026, 1.0783865284913943, 0.11303487689571717,
                        -0.8343466753906355, -6.906754778648555])
FIP_CENTERS = np.array([0.08230593607305912, 0.17734042553191487, 0.2917218543046358,
                        0.4465217391304345, 0.5945945945945944])
SI_EPS = 0.001


def dnc_strategy_index(trial):
    """SI = logit(VCR / (VCR + HPR + 0.001)), clipped to [0.001, 0.999]; equivalently
    log(VCR / (HPR + 0.001)) bounded at +-6.907. Rates use drift-active steps (all steps if none)."""
    steps = trial.get('steps', [])
    if not steps:
        return None
    selected = [s for s in steps if s.get('drift_row', 0) != 0] or steps
    vcr = sum(s.get('action', '').upper() in ('UP', 'DOWN') for s in selected) / len(selected)
    hpr = sum(s.get('action', '').upper() == 'RIGHT' for s in selected) / len(selected)
    share = min(1 - SI_EPS, max(SI_EPS, vcr / (vcr + hpr + SI_EPS)))
    return math.log(share / (1 - share))


def ctd_belief(trial):
    """Last Patient Risk Score entered before the ESI was finalized."""
    for step in reversed(trial.get('steps', [])):
        updates = step.get('BC_updates', [])
        if updates:
            return float(updates[-1]['ctx'])
    return None


def parse_esi(value):
    if isinstance(value, (int, float)) and 1 <= int(value) <= 5:
        return int(value)
    match = re.search(r'([1-5])', str(value))
    return int(match.group(1)) if match else None


def extract_trial(task, trial):
    """Return (B_C on 0-100, raw decision measure) or None if the pair is incomplete.
    Raw measure: CTD finalized ESI, FIP high-risk share H/(L+M+H), DNC strategy index."""
    if task == 'CTD':
        bc, esi = ctd_belief(trial), parse_esi((trial.get('finalized') or {}).get('ESI'))
        return None if bc is None or esi is None else (bc, float(esi))
    if task == 'FIP':
        report = trial.get('report') or {}
        bc = (report.get('contextual') or {}).get('risk')
        alloc = report.get('alloc') or {}
        try:
            low, mid, high = (float(alloc[k]) for k in ('L', 'M', 'H'))
        except (KeyError, TypeError, ValueError):
            return None
        total = low + mid + high
        return None if bc is None or total <= 0 else (float(bc), high / total)
    bc, si = trial.get('context_belief'), dnc_strategy_index(trial)
    return None if bc is None or si is None else (float(bc), si)


def assign_rd(task, raw):
    if task == 'CTD':
        return int(raw)
    centers = DNC_CENTERS if task == 'DNC' else FIP_CENTERS
    return int(np.argmin(np.abs(centers - raw)) + 1)


def load_human(data_dir=HUMAN_DATA):
    """Long table of retained human trials and a per-task retention audit.

    `participant` is the public cohort order (1..100); rows keep the file order, which is the
    cohort order, so participant resampling is reproducible."""
    cohort = pd.read_csv(data_dir / COHORT_FILE, dtype={'subject_id': str})
    order = dict(zip(cohort['subject_id'], cohort['cohort_order']))
    rows, audit = [], []
    for task, filename in HUMAN_FILES.items():
        records = pd.read_csv(data_dir / filename, dtype={'Subject ID': str})
        if len(records) != len(order) or set(records['Subject ID']) != set(order):
            raise RuntimeError(f'{task}: participants differ from {COHORT_FILE}')
        n_raw = n_kept = n_people = 0
        for subject, payload in zip(records['Subject ID'], records['Data (JSON)']):
            trials = json.loads(payload).get('trials', [])
            kept = 0
            for index, trial in enumerate(trials, start=1):
                pair = extract_trial(task, trial)
                if pair is None:
                    continue
                rows.append(dict(participant=int(order[subject]), task=task, trial=index,
                                 bc=pair[0], raw=pair[1], rd=assign_rd(task, pair[1])))
                kept += 1
            n_raw, n_kept, n_people = n_raw + len(trials), n_kept + kept, n_people + (kept > 0)
        audit.append(dict(task=task, released_participants=len(records), raw_trials=n_raw,
                          retained_participants=n_people, retained_trials=n_kept))
    return pd.DataFrame(rows), pd.DataFrame(audit)


def load_llm(main_dir=FRONTIER_MAIN, tasks=('DNC', 'CTD', 'FIP')):
    """Phase-2 records of the six frontier models (100 trials per model and task)."""
    rows = []
    for task in tasks:
        for model in MODELS:
            with open(main_dir / task / f'{task}_{model}_N100.json', encoding='utf-8') as fh:
                trials = json.load(fh)['trials']
            pairs = [extract_trial(task, t) for t in trials]
            if len(pairs) != 100 or any(p is None for p in pairs):
                raise RuntimeError(f'{task}/{model}: expected 100 complete trials')
            rows += [dict(model=model, task=task, trial=i, bc=bc, raw=raw, rd=assign_rd(task, raw))
                     for i, (bc, raw) in enumerate(pairs, start=1)]
    return pd.DataFrame(rows)
