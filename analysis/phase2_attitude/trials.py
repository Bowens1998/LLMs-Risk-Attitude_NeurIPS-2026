"""Load the Phase-2 trials (100 per model and task) and extract B_C and R_D."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from riskattitude.paths import EARLIER_MAIN, FRONTIER_MAIN, TASKS

from .ordinalization import assign_categories, strategy_index

# Model order fixes the order of random-number consumption in the bootstrap.
FRONTIER_MODELS = ('DeepSeekV3.2', 'Gemini3Pro', 'GPT5.2', 'Grok4', 'Qwen3Max', 'Sonnet4.5')
FRONTIER_LABELS = {
    'DeepSeekV3.2': 'DeepSeek V3.2', 'Gemini3Pro': 'Gemini 3 Pro', 'GPT5.2': 'GPT-5.2',
    'Grok4': 'Grok 4', 'Qwen3Max': 'Qwen3 Max', 'Sonnet4.5': 'Sonnet 4.5',
}
# Models that pass the Phase-1 reliability gate in all three tasks (task-average
# P1 <= 20% and P2 >= 80%); Grok 4 fails DNC and CTD.
RELIABLE_MODELS = ('DeepSeekV3.2', 'Gemini3Pro', 'GPT5.2', 'Qwen3Max', 'Sonnet4.5')
# Row order of the descriptive rank tables and figures.
DISPLAY_ORDER = ('Grok4', 'Sonnet4.5', 'DeepSeekV3.2', 'Gemini3Pro', 'Qwen3Max', 'GPT5.2')

EARLIER_MODELS = ('DeepSeekV3', 'Gemini2.5Pro', 'GPT4.1', 'Grok3', 'Qwen3', 'Sonnet4')
EARLIER_LABELS = {
    'DeepSeekV3': 'DeepSeek V3', 'Gemini2.5Pro': 'Gemini 2.5 Pro', 'GPT4.1': 'GPT-4.1',
    'Grok3': 'Grok 3', 'Qwen3': 'Qwen 3', 'Sonnet4': 'Sonnet 4',
}

ROW_ORDER_FILE = Path(__file__).with_name('reference_row_order.json')


def ctd_belief(trial: dict) -> float | None:
    """Patient Risk Score reported at the last observation step with an update."""
    for step in reversed(trial.get('steps', [])):
        updates = step.get('BC_updates') or []
        if updates and updates[-1].get('ctx') is not None:
            return float(updates[-1]['ctx'])
    return None


def parse_esi(value) -> int | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value) if 1 <= value <= 5 else None
    match = re.search(r'[1-5]', str(value))
    return int(match.group()) if match else None


def frontier_features(task: str, trial: dict) -> tuple[float | None, float | None]:
    """(B_C on 0-100, raw action index) for one frontier-model trial."""
    if task == 'DNC':
        bc = trial.get('context_belief')
        raw = strategy_index(trial.get('steps', []))
    elif task == 'CTD':
        bc = ctd_belief(trial)
        raw = parse_esi((trial.get('finalized') or {}).get('ESI'))
    else:
        report = trial.get('report') or {}
        bc = (report.get('contextual') or {}).get('risk')
        alloc = report.get('alloc') or {}
        total = sum(float(alloc.get(k, 0)) for k in ('L', 'M', 'H'))
        raw = float(alloc.get('H', 0)) / total if total > 0 else None
    if bc is None or raw is None:
        return None, None
    return float(bc), float(raw)


def load_frontier(reference_order: bool = True) -> pd.DataFrame:
    """One row per valid trial: task, model, trial, bc, raw, rd, out_of_support.

    With `reference_order`, cells listed in reference_row_order.json are reordered
    to the order used for the reference bootstrap.
    """
    order = json.loads(ROW_ORDER_FILE.read_text(encoding='utf-8')) if reference_order else {}
    frames = []
    for task in TASKS:
        for model in FRONTIER_MODELS:
            path = FRONTIER_MAIN / task / f'{task}_{model}_N100.json'
            trials = json.loads(path.read_text(encoding='utf-8'))['trials']
            rows = []
            for position, trial in enumerate(trials):
                bc, raw = frontier_features(task, trial)
                if bc is not None:
                    rows.append({'task': task, 'model': model,
                                 'trial': trial.get('index', position), 'bc': bc, 'raw': raw})
            cell = pd.DataFrame(rows)
            if model in order.get(task, {}):
                cell = cell.set_index('trial').loc[order[task][model]].reset_index()
            cell['rd'], cell['out_of_support'] = assign_categories(task, cell['raw'])
            frames.append(cell[['task', 'model', 'trial', 'bc', 'raw', 'rd', 'out_of_support']])
    return pd.concat(frames, ignore_index=True)


def earlier_belief(task: str, trial: dict) -> float | None:
    if task == 'DNC':
        value = trial.get('contextual_belief')
    elif task == 'FIP':
        value = ((trial.get('report') or {}).get('contextual') or {}).get('risk')
    else:
        value = ctd_belief(trial)
    return None if value is None else float(value)


def load_earlier() -> pd.DataFrame:
    """Earlier-generation models; R_D is the released `risk_decision` field."""
    rows = []
    for task in TASKS:
        for model in EARLIER_MODELS:
            path = EARLIER_MAIN / task / f'{task}_{model}_N100.json'
            trials = json.loads(path.read_text(encoding='utf-8'))['trials']
            for position, trial in enumerate(trials):
                bc = earlier_belief(task, trial)
                rd = trial.get('risk_decision')
                if bc is not None and rd is not None:
                    rows.append({'task': task, 'model': model, 'trial': position,
                                 'bc': bc, 'rd': int(rd)})
    return pd.DataFrame(rows)


def cells(df: pd.DataFrame, models, tasks=TASKS) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """{(task, model): (bc, rd)} in task-major, then model order."""
    out = {}
    for task in tasks:
        for model in models:
            g = df[(df.task == task) & (df.model == model)]
            out[task, model] = (g.bc.to_numpy(float), g.rd.to_numpy(int))
    return out
