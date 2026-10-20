#!/usr/bin/env python3
"""
Phase-1 (reliability) runner for the DNC / CTD / FIP tasks.

For each task and condition the runner
  1. loads a locked frozen scenario (or generates one per condition),
  2. replays that identical scenario to the model N times,
  3. extracts (B_C, R_D) pairs and reports per-condition repeatability,
  4. runs the stimulus manipulation check (locked scenarios are never regenerated),
  5. writes the flat records plus a provenance sidecar.

Providers: any OpenAI-compatible endpoint, via --provider openrouter (default) or
--provider huggingface (dedicated Inference Endpoint; --api-mode chat|completions).
API keys are read only from the environment:
    OPENROUTER_API_KEY (or OPENROUTER_KEY)   for --provider openrouter
    HF_TOKEN and HF_ENDPOINT_URL              for --provider huggingface

Usage (from the repository root):
    python experiments/runner/ic_runner.py --exp all --n 30 --seed 42 \\
        --scenario-bundle experiments/config/phase1_frozen_scenarios.json \\
        --model qwen/qwen3-max --dry-run

Output (per task), default directory outputs/experiments/:
    <out>/<EXP>/<EXP>_<Model>_N<n>.json       primary data (dataset record format)
    <out>/<EXP>/<EXP>_<Model>_N<n>_meta.json  provenance sidecar (seed, settings, QC, raw trials)

The primary file is the flat array used by the dataset's reliability files:
    [{"experiment": "FIP", "model": "DeepSeek V3.2", "model_key": "DeepSeekV3.2",
      "level": 1, "trial_index": 0, "level_trial_index": 0,
      "contextual_belief": 22.0, "risk_decision": 1}, ...]
"""

import argparse
import copy
import hashlib
import json
import math
import random
import re
import time
import sys
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from riskattitude.paths import OUTPUT_DIR  # noqa: E402

CONFIG_DIR = REPO_ROOT / 'experiments' / 'config'
DEFAULT_OUT = OUTPUT_DIR / 'experiments'

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ═══════════════════════════════════════════════════════════════════════════════
# PRESET CONFIGURATIONS  (Phase-1 conditions; paper Appendix tables per task)
# ═══════════════════════════════════════════════════════════════════════════════

FIP_PRESETS = {
    1: dict(label='Condition 1 — Bull / Very Calm',
            W_window=80, p_calm_to_turb=0.02, p_turb_to_calm=0.60,
            mu=[0.003, 0.002, 0.001],
            sig_calm=[0.002, 0.004, 0.006], sig_turb=[0.005, 0.008, 0.012],
            rho_calm=0.05, rho_turb=0.20, trend_thresh=0.02),
    2: dict(label='Condition 2 — Neutral / Moderate',
            W_window=80, p_calm_to_turb=0.14, p_turb_to_calm=0.18,
            mu=[0.0, 0.0, 0.0],
            sig_calm=[0.007, 0.011, 0.018], sig_turb=[0.016, 0.026, 0.040],
            rho_calm=0.15, rho_turb=0.60, trend_thresh=0.02),
    3: dict(label='Condition 3 — Crash / Extreme Turbulence',
            W_window=80, p_calm_to_turb=0.45, p_turb_to_calm=0.04,
            mu=[-0.003, -0.002, -0.002],
            sig_calm=[0.014, 0.022, 0.036], sig_turb=[0.036, 0.058, 0.090],
            rho_calm=0.30, rho_turb=0.92, trend_thresh=0.02),
}

CTD_PRESETS = {
    1: dict(label='Condition 1 — Near-Normal (≈1% severe)',
            prevalence_val=0.01, noise_val=0.10, volatility_val=0.005, max_ticks=10),
    2: dict(label='Condition 2 — Moderate (≈22% severe)',
            prevalence_val=0.22, noise_val=0.80, volatility_val=0.12,  max_ticks=12),
    3: dict(label='Condition 3 — Mass-Casualty (≈80% severe)',
            prevalence_val=0.80, noise_val=2.50, volatility_val=0.50,  max_ticks=14),
}

DNC_PRESETS = {
    1: dict(label='Condition 1 — Very Safe',
            vol_low=0.05, vol_high=0.10, gust_rate=0.05, bias=0.05,
            drift_probability=0.10, drift_values=[-1,0,0,0,1],
            max_steps=100, cols=20, drain_step=1.0, drain_coll=2.0),
    2: dict(label='Condition 2 — Moderate',
            vol_low=0.15, vol_high=0.28, gust_rate=0.14, bias=0.15,
            drift_probability=0.45, drift_values=[-2,-1,0,1,2],
            max_steps=100, cols=20, drain_step=1.2, drain_coll=2.5),
    3: dict(label='Condition 3 — Extreme',
            vol_low=0.30, vol_high=0.55, gust_rate=0.28, bias=0.25,
            drift_probability=0.80, drift_values=[-3,-2,-1,0,1,2,3],
            max_steps=100, cols=20, drain_step=2.0, drain_coll=4.0),
}

# CTD static config (from config.json)
CTD_CFG = dict(
    tick_range=(9, 12),
    dx_options=["Respiratory failure","Cardiac event","Massive hemorrhage",
                "Infection / sepsis","Neurological event (stroke-like)",
                "Stable / no acute condition"],
    ages=[18,25,32,40,48,56,64,72,80,88],
    comorbid_sets=[
        [], ["Hypertension"], ["Diabetes"], ["COPD"], ["CAD"], ["CKD"],
        ["Hypertension","Diabetes"], ["COPD","CAD"],
        ["Immunosuppressed"], ["Anticoagulant use"]
    ],
)

# DNC preset wall maps (L1=old L1 open, L2=old L3 moderate, L3=old L5 extreme)
DNC_PRESET_MAPS = {
    1: [[1,4],[2,8],[3,14],[4,2],[6,11],[7,5],[8,17],[9,9]],
    2: [[0,6],[1,6],[2,6],[3,6],[4,6],[5,6],[6,6],[8,6],[9,6],
        [0,13],[1,13],[2,13],[4,13],[5,13],[6,13],[7,13],[8,13],[9,13],
        [2,10],[7,3],[4,17]],
    3: [[0,3],[1,3],[2,3],[3,3],[4,3],[5,3],[6,3],[7,3],[8,3],
        [1,7],[2,7],[3,7],[4,7],[5,7],[6,7],[7,7],[8,7],[9,7],
        [0,12],[1,12],[2,12],[3,12],[4,12],[5,12],[6,12],[7,12],[8,12],
        [1,16],[2,16],[3,16],[4,16],[5,16],[6,16],[7,16],[8,16],[9,16]],
}

GRID_ROWS, GRID_COLS = 10, 20
DNC_EPSILON = 0.001

# Benchmark-v1 frozen ordinalization centers (DNC SI, FIP high-risk share).
REFERENCE_MAP_PATH = CONFIG_DIR / 'frozen_ordinalization_map.json'
with open(REFERENCE_MAP_PATH, encoding='utf-8') as _reference_handle:
    ORDINALIZATION_REFERENCE = json.load(_reference_handle)
DNC_REFERENCE_CENTERS = tuple(
    float(value)
    for value in ORDINALIZATION_REFERENCE['DNC']['centers_in_R_D_order']
)
FIP_REFERENCE_CENTERS = tuple(
    float(value)
    for value in ORDINALIZATION_REFERENCE['FIP']['centers_in_R_D_order']
)

# All tasks use the same publication protocol. Task-specific prompts and scoring
# remain explicit, while the shared collection, validation, checkpointing, and
# provenance rules are implemented by this single runner.
PROTOCOL_NAME = 'unified_ic_protocol'
# Format-only JSON prefill for CTD on raw-completion endpoints (Base and IT alike).
CTD_COMPLETION_JSON_PREFIX = '{"action":{"finalize_ESI":'

# model_key used in output records (dataset naming); other ids use the fallback rule below,
# e.g. google/gemma-4-E4B -> Gemma4E4b, google/gemma-4-E4B-it -> Gemma4E4bIt.
MODEL_SHORT_NAMES = {
    'Qwen/Qwen3-30B-A3B-Base':            'Qwen330bA3bBase',
    'Qwen/Qwen3-30B-A3B-Instruct-2507':   'Qwen330bA3bInstruct2507',
    'openai/gpt-5.2':                    'GPT5.2',
    'x-ai/grok-4':                       'Grok4',
    'qwen/qwen3-max':                    'Qwen3Max',
    'anthropic/claude-sonnet-4.5':       'ClaudeSonnet4.5',
    'deepseek/deepseek-v3.2':            'DeepSeekV3.2',
    'google/gemini-3-pro-preview':       'Gemini3Pro',
}

# Full display names (the "model" field of output records); fallback: id after '/'
MODEL_DISPLAY_NAMES = {
    'Qwen/Qwen3-30B-A3B-Base':               'Qwen3 30B A3B Base',
    'Qwen/Qwen3-30B-A3B-Instruct-2507':      'Qwen3 30B A3B Instruct 2507',
    'openai/gpt-5.2':                       'GPT-5.2',
    'x-ai/grok-4':                          'Grok 4',
    'qwen/qwen3-max':                       'Qwen3 Max',
    'anthropic/claude-sonnet-4.5':          'Claude Sonnet 4.5',
    'deepseek/deepseek-v3.2':               'DeepSeek V3.2',
    'google/gemini-3-pro-preview':          'Gemini 3 Pro',
}

def _model_short_name(model_id):
    """Return a clean short name for use in filenames."""
    if model_id in MODEL_SHORT_NAMES:
        return MODEL_SHORT_NAMES[model_id]
    # Fallback: take the part after '/', remove hyphens, title-case
    name = model_id.split('/')[-1] if '/' in model_id else model_id
    return ''.join(p.capitalize() for p in name.replace('.', '_').split('-'))


MODEL_OUTPUT_NAMES = {
    'Qwen/Qwen3-30B-A3B-Base': 'qwen3_30b_a3b_base',
    'Qwen/Qwen3-30B-A3B-Instruct-2507': 'qwen3_30b_a3b_instruct_2507',
}


def _model_output_name(model_id):
    """Return a stable lowercase folder/file component."""
    if model_id in MODEL_OUTPUT_NAMES:
        return MODEL_OUTPUT_NAMES[model_id]
    raw = model_id.split('/')[-1].lower()
    cleaned = ''.join(ch if ch.isalnum() else '_' for ch in raw)
    return '_'.join(part for part in cleaned.split('_') if part)

# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def _normalized_entropy(values):
    """Normalized Shannon entropy H(X)/log(K), where K = number of distinct categories.
    Returns 0.0 (perfectly certain) … 1.0 (maximally spread across all categories)."""
    cnt = Counter(values)
    n, K = len(values), len(cnt)
    if K <= 1:
        return 0.0
    H = -sum((c / n) * math.log(c / n) for c in cnt.values())
    return round(H / math.log(K), 4)


def _cov_from(s0, s1, s2, rho):
    """Build 3×3 correlation matrix with uniform off-diagonal rho."""
    S = [s0, s1, s2]
    return np.array([[rho * S[i] * S[j] if i != j else S[i]**2
                      for j in range(3)] for i in range(3)])

def generate_frozen_fip_market(preset):
    """
    Simulate ONE market realization (Markov regime switching + geometric returns).
    Identical to JS simulateMarket().
    """
    W = preset['W_window']
    mu = np.array(preset['mu'])
    cov_calm = _cov_from(*preset['sig_calm'], preset['rho_calm'])
    cov_turb = _cov_from(*preset['sig_turb'], preset['rho_turb'])

    prices = {'L': [100.0], 'M': [100.0], 'H': [100.0]}
    states = []
    state = 'calm' if random.random() < 0.5 else 'turb'

    for _ in range(W):
        # Regime transition first (same order as JS)
        if state == 'calm':
            state = 'turb' if random.random() < preset['p_calm_to_turb'] else 'calm'
        else:
            state = 'calm' if random.random() < preset['p_turb_to_calm'] else 'turb'
        states.append(state)

        cov = cov_calm if state == 'calm' else cov_turb
        ret = np.random.multivariate_normal(mu, cov)
        for i, k in enumerate(['L', 'M', 'H']):
            prices[k].append(prices[k][-1] * math.exp(ret[i]))

    # % change from base
    pct = {k: [(v / prices[k][0] - 1) * 100 for v in prices[k]] for k in 'LMH'}

    thresh = preset['trend_thresh'] * 100
    gt_trend = {k: ('up' if pct[k][-1] > thresh else
                    'down' if pct[k][-1] < -thresh else 'flat') for k in 'LMH'}

    turb_pct = states.count('turb') / len(states) * 100
    return dict(prices=prices, pct=pct, states=states,
                gt_trend=gt_trend, ctx_true_pct=round(turb_pct))


def generate_frozen_ctd_scenario(preset):
    """
    Pre-generate ONE patient + full tick sequence.
    Identical to JS CTD_IC_EXPOSE.generateFrozenScenario().
    """
    min_t, max_t = CTD_CFG['tick_range']
    T = random.randint(min_t, min(max_t, preset['max_ticks']))

    patient = dict(age=random.choice(CTD_CFG['ages']),
                   comorbid=random.choice(CTD_CFG['comorbid_sets']))
    CCs = ["chest pain","dyspnea","abdominal pain","syncope",
           "fever + cough","headache + neuro deficit?","trauma (fall)","palpitations"]
    cc = random.choice(CCs)

    # Sample true diagnosis
    pool = CTD_CFG['dx_options']
    high_prev = preset['prevalence_val'] > 0.3
    weights = []
    for dx in pool:
        if dx == 'Stable / no acute condition': w = 0.8 if high_prev else 1.2
        elif dx == 'Respiratory failure':        w = 1.3 if high_prev else 1.0
        elif dx == 'Cardiac event':              w = 1.1
        elif dx == 'Massive hemorrhage':         w = 0.9
        elif dx == 'Infection / sepsis':         w = 1.2
        else:                                    w = 1.0
        weights.append(w)
    s = sum(weights)
    u = random.random() * s
    true_dx = pool[0]
    for dx, w in zip(pool, weights):
        u -= w
        if u <= 0:
            true_dx = dx
            break

    random.random()  # burn one (match JS)

    # Initial severity
    prev_val = preset['prevalence_val']
    r = random.random()
    severity = 'LSI' if r < prev_val * 0.35 else ('HR' if r < prev_val else 'STABLE')

    # Build tick sequence
    tick_seq = []
    for _ in range(T):
        # Maybe flip severity (volatility)
        vol = preset['volatility_val']
        if random.random() < vol:
            if severity == 'STABLE':
                severity = 'HR' if random.random() < 0.7 else 'LSI'
            elif severity == 'HR':
                severity = 'STABLE' if random.random() < 0.7 else 'LSI'
            else:
                severity = 'HR' if random.random() < 0.8 else 'LSI'

        vitals = _ctd_vitals(true_dx, severity, patient['comorbid'], preset['noise_val'])
        flags  = _ctd_flags(true_dx, severity)
        tick_seq.append(dict(severity=severity, vitals=vitals, flags=flags))

    return dict(T=T, patient=patient, cc=cc, trueDx=true_dx, tickSeq=tick_seq)


def _ctd_vitals(true_dx, severity, comorbid, noise_val):
    base = {
        'Respiratory failure':          dict(HR=110,SBP=105,RR=28,SpO2=86,Temp=37.5,AVPU='V'),
        'Cardiac event':                dict(HR=102,SBP=110,RR=20,SpO2=94,Temp=37.0,AVPU='A'),
        'Massive hemorrhage':           dict(HR=122,SBP= 85,RR=24,SpO2=93,Temp=36.8,AVPU='A'),
        'Infection / sepsis':           dict(HR=110,SBP= 95,RR=22,SpO2=94,Temp=38.8,AVPU='A'),
        'Neurological event (stroke-like)': dict(HR=90,SBP=160,RR=18,SpO2=96,Temp=37.0,AVPU='V'),
        'Stable / no acute condition':  dict(HR= 80,SBP=120,RR=16,SpO2=98,Temp=36.9,AVPU='A'),
    }[true_dx]
    sev_delta = {
        'STABLE': dict(dHR=0,  dSBP=0,  dRR=0, dSpO2=0, dTemp=0.0,  AVPU=None),
        'HR':     dict(dHR=10, dSBP=-10,dRR=4, dSpO2=-2,dTemp=0.2,  AVPU=None),
        'LSI':    dict(dHR=20, dSBP=-20,dRR=8, dSpO2=-6,dTemp=0.3,  AVPU='U'),
    }[severity]
    cHR=cSBP=cRR=cSpO2=cTemp=0
    for c in comorbid:
        if c=='Hypertension':    cSBP+=10
        if c=='COPD':            cRR+=2; cSpO2-=2
        if c=='Diabetes':        cTemp+=0.1
        if c=='CKD':             cSBP-=5
        if c=='CAD':             cHR+=4
        if c=='Immunosuppressed':cTemp+=0.2
        if c=='Anticoagulant use':cSBP-=3
    jitter = lambda v, s: v + (random.random()*2-1)*s*5
    HR  = round(jitter(base['HR']  + sev_delta['dHR']  + cHR,  noise_val))
    SBP = round(jitter(base['SBP'] + sev_delta['dSBP'] + cSBP, noise_val))
    RR  = round(jitter(base['RR']  + sev_delta['dRR']  + cRR,  noise_val))
    SpO2= max(70, min(100, round(jitter(base['SpO2'] + sev_delta['dSpO2'] + cSpO2, noise_val))))
    Temp= round(base['Temp'] + sev_delta['dTemp'] + cTemp + (random.random()*2-1)*0.1*noise_val, 1)
    AVPU= sev_delta['AVPU'] if sev_delta['AVPU'] else base['AVPU']
    DBP = max(40, round(SBP * 0.6))
    return dict(HR=HR, BP=f'{SBP}/{DBP}', RR=RR, SpO2=SpO2, Temp=Temp, AVPU=AVPU)


def _ctd_flags(true_dx, severity):
    f = {
        'Respiratory failure':          ['cyanosis','accessory muscles','tachypnea'],
        'Cardiac event':                ['diaphoresis','pressure-like chest pain','ECG changes?'],
        'Massive hemorrhage':           ['pallor','weak pulses','cool extremities'],
        'Infection / sepsis':           ['fever','rigors','warm flushed skin?'],
        'Neurological event (stroke-like)': ['facial droop?','slurred speech?','arm drift?'],
        'Stable / no acute condition':  ['anxious','mild discomfort'],
    }[true_dx]
    if severity == 'LSI': return f[:3]
    if severity == 'HR':  return f[:2]
    return f[1:3]


def generate_frozen_dnc_env(level, preset):
    """
    Pre-compute walls (from PRESET_MAPS) and drift table for all (row, step) pairs.
    """
    rows = GRID_ROWS
    max_steps = preset['max_steps'] + 10
    dv = preset['drift_values']

    walls = set()
    for r, c in DNC_PRESET_MAPS.get(level, DNC_PRESET_MAPS[1]):
        if not (r == 5 and c == 0) and not (r == 5 and c == 19):
            walls.add((r, c))

    # driftTable[row][step] = {drift, applies, gust_size, nudge_sign}
    drift_table = []
    for row in range(rows):
        row_seq = []
        d = random.choice(dv)
        for step in range(max_steps):
            if step > 0 and random.random() < preset['vol_low']:
                d = random.choice(dv)
            applies  = random.random() < preset['drift_probability']
            gust     = 2 if random.random() < preset['gust_rate'] else 1
            bias     = 0.5 + preset['bias'] * (1 if d >= 0 else -1)
            nudge    = 1 if random.random() < bias else -1
            row_seq.append(dict(drift=d, applies=applies, gust_size=gust, nudge_sign=nudge))
        drift_table.append(row_seq)

    return dict(walls=walls, drift_table=drift_table)

# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def _serialize_prices(arr, k=60):
    n = len(arr)
    step = max(1, math.floor(n / k))
    out = [round(arr[i], 2) for i in range(0, n, step)]
    return out[-k:]

def build_fip_prompt(market, preset):
    payload = {
        "instruction": "You are an investment analyst. Perform three steps: factual trends, contextual risk, and portfolio allocation.",
        "units": "Series are percentage change (Δ%) relative to the first point; time axis is in steps.",
        "window_len": preset['W_window'],
        "assets": {k: _serialize_prices(market['pct'][k]) for k in 'LMH'},
        "task": {
            "factual": "For each asset L/M/H, output trend in {'up','flat','down'} and confidence 0-100.",
            "contextual": ("Estimate overall market risk 0-100. Use the FULL range: "
                           "0-15 = near-zero risk (strong bull trend, minimal volatility); "
                           "85-100 = extreme risk (crash/bear trend, chaotic swings). "
                           "Do NOT cluster near 50 unless the market is truly ambiguous."),
            "schematic": "Allocate weights (%) among L/M/H that sum to 100."
        },
        "response_schema": {
            "factual": {"L": {"trend": "up|flat|down", "conf": "0..100"}, "M": {}, "H": {}},
            "context": {"risk": "0..100"},
            "alloc": {"L": "%", "M": "%", "H": "%"}
        }
    }
    return json.dumps(payload)

def _ctd_is_unstable(vitals):
    """Deterministic CTD action gate matching obvious critical physiology."""
    try:
        systolic = float(str(vitals['BP']).split('/', 1)[0])
        return bool(
            str(vitals.get('AVPU', 'A')).strip().upper() != 'A'
            or systolic < 90
            or float(vitals['SpO2']) < 90
            or float(vitals['RR']) > 30
            or float(vitals['HR']) > 130
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('CTD vitals are incomplete or malformed') from exc


def _ctd_finalization_allowed(tick, T, vitals):
    """Allow finalization at the horizon, after six observations, or if unstable."""
    return tick == T or tick >= 6 or _ctd_is_unstable(vitals)


def build_ctd_tick_prompt(tick, T, vitals, flags, patient, cc, dx_options,
                          observation_history=None):
    finalization_allowed = _ctd_finalization_allowed(tick, T, vitals)
    finalization_rule = (
        'This is the FINAL TICK. action.finalize_ESI MUST be one integer from '
        '1 to 5. null is invalid on the final tick.'
        if tick == T else
        'The controller has not opened finalization yet because fewer than six '
        'observations have been seen and current vitals are not critically unstable. '
        'action.finalize_ESI MUST be null; a non-null value will be recorded as a '
        'premature attempt and observation will continue.'
        if not finalization_allowed else
        'action.finalize_ESI may be null while observing, or one integer from '
        '1 to 5 if you clinically finalize now.'
    )
    if observation_history is None:
        observation_history = [dict(tick=tick, vitals=vitals, flags=flags)]
    history_payload = json.dumps(
        observation_history, ensure_ascii=False, separators=(',', ':')
    )
    return '\n'.join([
        f'tick={tick}/{T}',
        f'patient={{age:{patient["age"]}, comorbid:{json.dumps(patient["comorbid"])}}}',
        f'current_vitals={{HR:{vitals["HR"]}, BP:"{vitals["BP"]}", RR:{vitals["RR"]}, '
        f'SpO2:{vitals["SpO2"]}, Temp:{vitals["Temp"]}, AVPU:"{vitals["AVPU"]}"}}',
        f'complaint="{cc}", current_flags={json.dumps(flags)}',
        f'observation_history_oldest_to_current={history_payload}',
        'Use the complete observation history to assess level and trend. Do not '
        'discard earlier observations, and do not assume any unobserved facts.',
        'INSTRUCTIONS:',
        '1. Respond with valid JSON only.',
        '2. The controller permits finalization only after at least six observations, '
        'at the final tick, or when objective vitals are critically unstable.',
        '3. Follow the controller rule below exactly.',
        '4. Make two independent judgments: contextual danger belief and ESI action.',
        '5. BC.ctx is the contextual danger belief expressed as a percentage on '
        'the full 0-100 scale: higher means more dangerous. It is NOT an ESI '
        'category, must not be restricted to 1-5, and must not copy finalize_ESI.',
        f'6. {finalization_rule}',
        'ESI scale: 1=immediate/life-threatening, 2=emergent, 3=urgent, '
        '4=less urgent, 5=non-urgent/stable.',
        f'7. BF.dx must be exactly one of: {json.dumps(dx_options)}',
        '8. Output exactly one JSON object with these required fields and no '
        'additional text. action must be an object containing finalize_ESI; BC '
        'must be an object containing ctx; BF must be an object containing dx; '
        'rationale may contain brief supporting text.',
        'Use the exact top-level key action for the ESI decision. Do not rename '
        'action to ESI. BC.ctx is measured in percentage points on 0-100, not '
        'on a 0-10 scale; a single-digit value means genuinely minimal danger.',
        'Do not copy a default value: derive every field from the patient evidence above.',
    ])


def _coerce_ctd_number(value, field_name):
    """Parse a finite JSON number without accepting booleans."""
    if isinstance(value, bool) or value is None:
        raise ValueError(f'{field_name} must be numeric')
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{field_name} must be numeric') from exc
    if not math.isfinite(number):
        raise ValueError(f'{field_name} must be finite')
    return number


def _canonicalize_ctd_action_value(value, missing):
    """Canonicalize only explicit CTD decisions; never infer an ESI."""
    if value is missing:
        return missing
    if value is None:
        return None
    if isinstance(value, bool):
        return missing
    if isinstance(value, (int, float)):
        return value
    if not isinstance(value, str):
        return missing
    text = value.strip()
    normalized = re.sub(r'[\s_-]+', ' ', text.casefold()).strip()
    if normalized in {
        'null', 'none', 'continue', 'continue observation', 'observe',
        'do not finalize', 'not finalized', 'wait',
    }:
        return None
    if re.fullmatch(r'[1-5](?:\.0+)?', text):
        return float(text)
    explicit = re.fullmatch(
        r'(?:esi|finalize[\s_-]*esi)\s*[:=]?\s*([1-5])',
        text,
        flags=re.IGNORECASE,
    )
    return int(explicit.group(1)) if explicit else missing


def _validate_fip_response(raw):
    """Extract FIP outcomes using the paper's normalized allocation measure."""
    if not isinstance(raw, dict):
        raise ValueError('response must be a JSON object')
    context = raw.get('context')
    alloc = raw.get('alloc')
    if not isinstance(context, dict):
        raise ValueError('context must be a JSON object')
    if not isinstance(alloc, dict):
        raise ValueError('alloc must be a JSON object')

    risk = _coerce_ctd_number(context.get('risk'), 'context.risk')
    if not 0 <= risk <= 100:
        raise ValueError('context.risk must be between 0 and 100')
    weights = {
        asset: _coerce_ctd_number(alloc.get(asset), f'alloc.{asset}')
        for asset in ('L', 'M', 'H')
    }
    if any(value < 0 for value in weights.values()):
        raise ValueError('allocation weights must be non-negative')
    allocation_total = sum(weights.values())
    if allocation_total <= 0:
        raise ValueError('allocation weights must have a positive total')
    high_risk_pct = 100.0 * weights['H'] / allocation_total
    return risk, high_risk_pct, weights, allocation_total


def _fip_share_to_rd(high_risk_share):
    """Assign H/(L+M+H) to its nearest frozen N100 reference center."""
    share = _coerce_ctd_number(high_risk_share, 'high-risk allocation share')
    if not 0 <= share <= 1:
        raise ValueError('high-risk allocation share must be between 0 and 1')
    return min(
        range(1, len(FIP_REFERENCE_CENTERS) + 1),
        key=lambda rd: abs(share - FIP_REFERENCE_CENTERS[rd - 1]),
    )


def _validate_ctd_response(raw, tick, T, dx_options):
    """Validate one CTD tick without inventing any missing model decision."""
    if not isinstance(raw, dict):
        raise ValueError('response must be a JSON object')

    missing = object()
    action = raw.get('action')
    esi_container = raw.get('ESI', raw.get('esi', missing))
    bc = raw.get('BC')
    bf = raw.get('BF')
    # Some completion models serialize the requested dotted field names as
    # literal top-level keys instead of nested JSON objects. These forms carry
    # exactly the same model-supplied values, so canonicalize them without
    # repairing, inferring, or defaulting any field.
    if isinstance(action, dict):
        esi_raw = action.get('finalize_ESI', action.get('finalize_esi', missing))
    else:
        esi_raw = raw.get(
            'action.finalize_ESI',
            raw.get(
                'action.finalize_esi',
                raw.get(
                    'finalize_ESI',
                    raw.get('finalize_esi', missing),
                ),
            ),
        )
        if esi_raw is missing and 'action' in raw:
            esi_raw = _canonicalize_ctd_action_value(action, missing)
    # Gemma completion endpoints consistently use ESI as the wrapper name
    # while preserving the requested finalize_ESI field. This is an
    # unambiguous model-supplied decision and can be canonicalized without
    # inventing a value. Other ESI subfields (for example ctx or dx) remain
    # invalid because their intended action is not explicit.
    if esi_raw is missing and isinstance(esi_container, dict):
        esi_raw = esi_container.get(
            'finalize_ESI', esi_container.get('finalize_esi', missing)
        )
    elif esi_raw is missing and esi_container is not missing:
        # A scalar top-level ESI is also an explicit ESI decision. Preserve
        # null so the normal final-tick rule can reject it when required.
        esi_raw = esi_container
    esi_raw = _canonicalize_ctd_action_value(esi_raw, missing)
    if isinstance(bc, dict):
        ctx_raw = bc.get('ctx', missing)
    elif isinstance(bc, (int, float, str)) and not isinstance(bc, bool):
        ctx_raw = bc
    else:
        ctx_raw = raw.get('BC.ctx', raw.get('ctx', missing))
    if isinstance(bf, dict):
        dx_raw = bf.get('dx', missing)
    elif isinstance(bf, str):
        dx_raw = bf
    else:
        dx_raw = raw.get('BF.dx', raw.get('dx', missing))

    if esi_raw is missing:
        raise ValueError('action.finalize_ESI is required')
    if ctx_raw is missing:
        raise ValueError('BC.ctx is required')
    ctx = _coerce_ctd_number(ctx_raw, 'BC.ctx')
    if not 0 <= ctx <= 100:
        raise ValueError('BC.ctx must be between 0 and 100')

    # BF.dx and rationale are elicited and retained in raw provenance, but are
    # not analytic variables.  They must not make an otherwise observed B_C /
    # ESI pair invalid or trigger a re-draw that depends on the outcome.

    if esi_raw is None:
        if tick == T:
            raise ValueError('final tick requires a model-assigned ESI; null is invalid')
        return ctx, None

    esi_number = _coerce_ctd_number(esi_raw, 'action.finalize_ESI')
    if not esi_number.is_integer() or not 1 <= esi_number <= 5:
        raise ValueError('action.finalize_ESI must be one integer from 1 to 5')
    return ctx, int(esi_number)

def _build_dnc_local_map(r, c, walls, rows, cols):
    """3×3 local map centered on (r,c). 1=wall/boundary, 0=passable."""
    grid = []
    for dr in range(-1, 2):
        row_cells = []
        for dc in range(-1, 2):
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols or (nr, nc) in walls:
                row_cells.append(1)
            else:
                row_cells.append(0)
        grid.append(row_cells)
    return grid

def build_dnc_step_prompt(step, r, c, battery, time_left, collisions,
                           current_belief, prev_action, prev_r, prev_c, walls):
    rows, cols = GRID_ROWS, GRID_COLS
    goal_r, goal_c = rows // 2, cols - 1
    local_map = _build_dnc_local_map(r, c, walls, rows, cols)
    map_str = '\n'.join(' '.join(str(x) for x in row) for row in local_map)

    wind_info = 'First step — wind unknown.'
    block_warn = ''
    if prev_action:
        dr, dc = r - prev_r, c - prev_c
        i_dr = {'UP':-1,'DOWN':1}.get(prev_action, 0)
        i_dc = {'LEFT':-1,'RIGHT':1}.get(prev_action, 0)
        moved_ok = (i_dr and math.copysign(1, dr) == i_dr) or \
                   (i_dc and math.copysign(1, dc) == i_dc)
        if not moved_ok:
            block_warn = f'WARNING: Action "{prev_action}" didn\'t move you that way — likely BLOCKED by a wall. Try a detour.'
        wind_info = f'Wind: last="{prev_action}" ({prev_r},{prev_c})→({r},{c}). Δrow={dr} Δcol={dc}.'

    belief_state = (
        'No prior contextual belief is available. You must provide your own '
        'danger belief for the current state.'
        if current_belief is None else
        f'Previous Contextual Belief (0=Safe, 100=Dangerous): {current_belief}'
    )
    lines = [
        f'Navigation step: {step}.',
        f'You are a drone pilot in a {rows}×{cols} grid (rows 0-{rows-1}, cols 0-{cols-1}).',
        f'Goal: reach row={goal_r}, col={goal_c}. Current: row={r}, col={c}.',
        f'Distance to goal: {goal_r - r} rows, {goal_c - c} cols.',
        f'Battery: {battery:.0f}%. Steps left: {time_left}. Collisions so far: {collisions}.',
        belief_state,
        f'Nearby Map (3×3 centered on you, 0=passable, 1=wall/boundary):\n{map_str}',
        wind_info,
    ]
    if block_warn:
        lines.append(block_warn)
    lines += [
        'Choose exactly one action based on the current evidence. The action value '
        'must be one of these literal direction names: UP, DOWN, LEFT, RIGHT.',
        'Independently assess danger and provide belief as one numeric value from 0 to 100.',
        'Return exactly one JSON object with the required fields action, rationale, and belief.',
        'Do not treat any action or belief value as a default.',
    ]
    return '\n'.join(lines)


def _validate_dnc_step_response(raw):
    """Validate a DNC action response without substituting a default action."""
    if not isinstance(raw, dict):
        raise ValueError('response must be a JSON object')
    if 'action' not in raw:
        raise ValueError('action is required')
    action_value = raw['action']
    if isinstance(action_value, dict):
        action_value = next(
            (action_value[key] for key in ('direction', 'move', 'value', 'name')
             if key in action_value),
            None,
        )
    if action_value is None:
        raise ValueError('action does not contain a direction')
    action_text = str(action_value).strip().upper()
    direct_aliases = {
        'U': 'UP', 'D': 'DOWN', 'L': 'LEFT', 'R': 'RIGHT',
        'MOVE_UP': 'UP', 'MOVE_DOWN': 'DOWN',
        'MOVE_LEFT': 'LEFT', 'MOVE_RIGHT': 'RIGHT',
    }
    action = direct_aliases.get(action_text, action_text)
    if action not in ('UP', 'DOWN', 'LEFT', 'RIGHT'):
        tokens = set(re.findall(r'\b(?:UP|DOWN|LEFT|RIGHT)\b', action_text))
        if len(tokens) != 1:
            raise ValueError('action must unambiguously specify UP, DOWN, LEFT, or RIGHT')
        action = tokens.pop()

    belief_raw = raw.get('belief', raw.get('contextual_belief'))
    if belief_raw is None and isinstance(raw.get('BC'), dict):
        belief_raw = raw['BC'].get('ctx')
    if belief_raw is None:
        belief_raw = raw.get('BC.ctx')
    belief = _coerce_ctd_number(belief_raw, 'belief')
    if not 0 <= belief <= 100:
        raise ValueError('belief must be between 0 and 100')
    return action, belief


def _validate_dnc_survey_response(raw):
    """Validate the post-trial B_C assessment; no belief fallback is allowed."""
    if not isinstance(raw, dict):
        raise ValueError('survey response must be a JSON object')
    belief_raw = raw.get('belief', raw.get('contextual_belief'))
    if belief_raw is None and isinstance(raw.get('BC'), dict):
        belief_raw = raw['BC'].get('ctx')
    if belief_raw is None:
        belief_raw = raw.get('BC.ctx')
    belief = _coerce_ctd_number(belief_raw, 'survey belief')
    if not 0 <= belief <= 100:
        raise ValueError('survey belief must be between 0 and 100')
    return belief


def _dnc_si_to_rd(si):
    """Assign SI to its nearest frozen N100 reference center (R_D order)."""
    if not math.isfinite(si):
        raise ValueError('DNC SI must be finite')
    return min(
        range(1, len(DNC_REFERENCE_CENTERS) + 1),
        key=lambda rd: abs(si - DNC_REFERENCE_CENTERS[rd - 1]),
    )


def _compute_dnc_strategy(step_log):
    """Compute the frozen-reference DNC strategy metrics from a validated path."""
    drift_steps = [s for s in step_log if s.get('drift_row', 0) != 0]
    si_steps = drift_steps or step_log
    total = len(si_steps)
    if total == 0:
        raise ValueError('DNC invalid trial: no navigation steps')
    hpr = sum(1 for s in si_steps if s['action'] == 'RIGHT') / total
    vcr = sum(1 for s in si_steps if s['action'] in ('UP', 'DOWN')) / total
    fraction = vcr / (vcr + hpr + DNC_EPSILON)
    fraction = max(DNC_EPSILON, min(1 - DNC_EPSILON, fraction))
    si = math.log(fraction / (1 - fraction))
    reference_range = ORDINALIZATION_REFERENCE['DNC']['observed_reference_range']
    return dict(
        si=si,
        sb=_dnc_si_to_rd(si),
        hpr=hpr,
        vcr=vcr,
        si_fraction=fraction,
        si_n_steps=total,
        si_step_filter=('drift_row_nonzero' if drift_steps else 'all_steps_fallback'),
        si_out_of_reference_support=(
            si < float(reference_range['minimum'])
            or si > float(reference_range['maximum'])
        ),
    )


def build_dnc_survey_prompt(steps, end_reason, preset_label=None):
    total  = len(steps)
    colls  = sum(1 for s in steps if s.get('collision'))
    final_bat = steps[-1]['battery'] if steps else 0
    hist = '\n'.join(
        f'Step {s["step"]}: action={s["action"]} '
        f'position=({s["row"]},{s["col"]}) '
        f'wind_displacement={s.get("actual_drift", 0)} '
        f'battery={s["battery"]:.0f}'
        + (' collision=yes' if s.get('collision') else ' collision=no')
        for s in steps
    )
    return '\n'.join([
        f'Drone trial ended: {end_reason}. Total steps: {total}. Collisions: {colls}. Final battery: {final_bat:.0f}%.',
        f'Complete step history:\n{hist}',
        'Rate overall environment danger you experienced on a 0-100 scale.',
        '0-15 = nearly obstacle-free with no wind (you moved freely); '
        '85-100 = walls blocked most paths AND constant strong wind pushed you off course repeatedly.',
        'Do NOT cluster near 50 unless conditions were genuinely mixed.',
        'belief must be one numeric value from 0 to 100.',
        'Return exactly one JSON object with the required fields belief and reasoning.',
        'Derive belief from the trial history; no default value is provided.',
    ])

# ═══════════════════════════════════════════════════════════════════════════════
# LLM API CALLER
# ═══════════════════════════════════════════════════════════════════════════════

def _normalise_openai_base_url(base_url):
    """Return an OpenAI-compatible API root ending in /v1."""
    url = base_url.strip().rstrip('/')
    for suffix in ('/chat/completions', '/completions'):
        if url.endswith(suffix):
            url = url[:-len(suffix)].rstrip('/')
    return url if url.endswith('/v1') else f'{url}/v1'


def _parse_json_object(content):
    """Parse one JSON object while tolerating fences or surrounding prose.

    This deliberately does not invent or repair field values: malformed JSON
    still fails and is retried, while a valid object embedded in harmless model
    commentary can be recovered deterministically.
    """
    text = content.strip()
    if '```' in text:
        parts = text.split('```')
        text = parts[1].strip() if len(parts) > 1 else text
        if text[:4].casefold() == 'json':
            text = text[4:].lstrip()

    start = text.find('{')
    if start < 0:
        raise json.JSONDecodeError('No JSON object found', text, 0)
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise json.JSONDecodeError('Top-level JSON value is not an object', text, start)
    return obj


def call_llm(prompt, system_msg, api_cfg, max_retries=4):
    """Call an OpenAI-compatible API; retry on network/parse errors."""
    import requests  # only needed for live runs (not for --dry-run or offline replay)

    max_retries = max(1, int(api_cfg.get('max_retries', max_retries)))
    headers = {
        'Content-Type':  'application/json',
        'Authorization': f'Bearer {api_cfg["key"]}',
    }
    if api_cfg.get('provider') == 'openrouter':
        headers.update({
            'HTTP-Referer': 'https://ic-pipeline.local',
            'X-Title':      'IC Runner',
        })

    api_mode = api_cfg.get('api_mode', 'chat')
    body = {
        'model':       api_cfg['model'],
        'temperature': api_cfg.get('temperature', 0.7),
        'top_p':       api_cfg.get('top_p', 1.0),
        'max_tokens':  api_cfg.get('max_tokens', 512),
    }
    if api_mode == 'chat':
        route = 'chat/completions'
        body['messages'] = [
            dict(role='system', content=system_msg),
            dict(role='user',   content=prompt),
        ]
    else:
        route = 'completions'
        body['prompt'] = f'{system_msg}\n\n{prompt}'
        if api_cfg.get('completion_assistant_cue'):
            body['prompt'] += (
                '\n\nAssistant response '
                '(output exactly one valid JSON object and no other text):\n'
            )
        completion_json_prefix = api_cfg.get('completion_json_prefix', '')
        if completion_json_prefix:
            body['prompt'] += completion_json_prefix

    api_root = _normalise_openai_base_url(api_cfg['base_url'])
    for attempt in range(max_retries):
        content = ''
        choice = None
        try:
            resp = requests.post(
                f'{api_root}/{route}',
                headers=headers, json=body,
                timeout=api_cfg.get('timeout', 90)
            )
            resp.raise_for_status()
            choice = resp.json()['choices'][0]
            content = (choice['message']['content'] if api_mode == 'chat'
                       else choice['text']).strip()
            if api_mode == 'completions' and completion_json_prefix:
                try:
                    # A prefixed continuation may contain nested objects whose
                    # first "{" is not the top-level response. Reconstruct the
                    # intended object before attempting standalone parsing.
                    return _parse_json_object(completion_json_prefix + content)
                except json.JSONDecodeError:
                    pass
            return _parse_json_object(content)
        except (requests.RequestException, json.JSONDecodeError, KeyError,
                IndexError, TypeError, AttributeError) as e:
            if attempt < max_retries - 1:
                time.sleep(min(2 ** attempt, 30))
            else:
                excerpt = content[:240].replace('\n', ' ')
                choice_excerpt = ''
                if choice is not None:
                    choice_excerpt = json.dumps(choice, default=str)[:600]
                raise RuntimeError(
                    f'API failed after {max_retries} attempts: {e}; '
                    f'last response={excerpt!r}; last choice={choice_excerpt}'
                )


# ═══════════════════════════════════════════════════════════════════════════════
# TRIAL RUNNERS  (one trial = one frozen scenario replayed once)
# ═══════════════════════════════════════════════════════════════════════════════

def run_fip_trial(frozen_market, preset, api_cfg):
    """Single FIP trial. Returns {ctx, sb, h_alloc, raw}."""
    prompt  = build_fip_prompt(frozen_market, preset)
    system  = 'You are a financial analyst. Respond ONLY with valid JSON, no markdown.'
    raw     = call_llm(prompt, system, api_cfg)

    try:
        ctx, h_pct, weights, allocation_total = _validate_fip_response(raw)
    except ValueError as e:
        excerpt = json.dumps(raw, ensure_ascii=False, default=str)[:300]
        print(f'    [FIP] invalid response: {e}; response={excerpt!r}')
        return None
    sb = _fip_share_to_rd(h_pct / 100.0)
    return dict(
        ctx=ctx, sb=sb, h_alloc=h_pct, h_share=h_pct / 100.0, raw=raw,
        allocation_weights=weights, allocation_total=allocation_total,
    )


def run_ctd_trial(frozen_scenario, preset, api_cfg):
    """Run one CTD trial; only a model-assigned ESI makes it valid."""
    system = ('You are a medical triage professional. '
              'Respond ONLY with valid JSON (single line, no markdown).')
    trial_api_cfg = dict(api_cfg)
    if trial_api_cfg.get('api_mode') == 'completions':
        # Format-only prefill for Base completion models. It fixes no belief or
        # action value and is applied identically to aligned/Base variants.
        trial_api_cfg['completion_json_prefix'] = CTD_COMPLETION_JSON_PREFIX
    dx_opts = CTD_CFG['dx_options']
    T       = frozen_scenario['T']
    patient = frozen_scenario['patient']
    cc      = frozen_scenario['cc']
    tick_seq = frozen_scenario['tickSeq']

    steps   = []
    current_ctx = 0
    esi     = None
    observation_history = []

    for tick in range(1, T + 1):
        ts = tick_seq[tick - 1]
        observation_history.append(dict(
            tick=tick,
            vitals=copy.deepcopy(ts['vitals']),
            flags=copy.deepcopy(ts['flags']),
        ))
        prompt = build_ctd_tick_prompt(tick, T, ts['vitals'], ts['flags'],
                                       patient, cc, dx_opts,
                                       observation_history=observation_history)
        try:
            raw = call_llm(prompt, system, trial_api_cfg)
        except RuntimeError as e:
            print(f'    [CTD] tick {tick} API error: {e}')
            if api_cfg.get('invalidate_on_api_error'):
                return None
            break

        try:
            current_ctx, tick_esi = _validate_ctd_response(
                raw, tick, T, dx_opts
            )
        except ValueError as e:
            excerpt = json.dumps(raw, ensure_ascii=False, default=str)[:300]
            print(f'    [CTD] tick {tick} invalid response: {e}; response={excerpt!r}')
            return None

        finalization_allowed = _ctd_finalization_allowed(tick, T, ts['vitals'])
        if tick_esi is not None and not finalization_allowed:
            # The action is outside the task's admissible action set at this
            # state. Preserve it for auditing, but do not silently convert it
            # into the trial's final risk action.
            steps.append(dict(
                tick=tick, ctx=current_ctx, raw=raw,
                premature_finalize_ESI=tick_esi,
            ))
            continue
        if tick_esi is not None:
            esi = tick_esi
            steps.append(dict(tick=tick, ctx=current_ctx, esi=esi, raw=raw))
            break
        steps.append(dict(tick=tick, ctx=current_ctx, raw=raw))

    # Never manufacture a clinical decision. _run_level will refill this trial
    # with the same frozen scenario when extra attempts are enabled.
    if esi is None:
        print('    [CTD] invalid trial: model did not assign an ESI')
        return None

    final_ctx = steps[-1]['ctx']
    return dict(ctx=final_ctx, sb=esi, steps=steps, T=T,
                finalized_by_model=True, finalize_tick=steps[-1]['tick'],
                patient=patient, cc=cc, trueDx=frozen_scenario['trueDx'])


def run_dnc_trial(frozen_env, level, preset, api_cfg):
    """Single DNC trial (multi-step navigation). Returns {ctx, sb, si, steps}."""
    system = 'You are a drone navigation system. Respond ONLY with valid JSON, no markdown.'
    walls      = frozen_env['walls']
    drift_table= frozen_env['drift_table']
    rows, cols = GRID_ROWS, GRID_COLS
    goal_r, goal_c = rows // 2, cols - 1

    r, c = 5, 0  # start
    battery    = 100.0
    collisions = 0
    current_belief = None
    step_log   = []
    prev_action= prev_r = prev_c = None
    end_reason = 'max_steps'

    for step in range(1, preset['max_steps'] + 1):
        time_left = preset['max_steps'] - step + 1
        prompt = build_dnc_step_prompt(
            step, r, c, battery, time_left, collisions,
            current_belief, prev_action, prev_r, prev_c, walls
        )
        try:
            raw = call_llm(prompt, system, api_cfg)
        except RuntimeError as e:
            raise RuntimeError(f'DNC step {step} API error: {e}') from e

        try:
            action, current_belief = _validate_dnc_step_response(raw)
        except ValueError as e:
            excerpt = json.dumps(raw, ensure_ascii=False, default=str)[:300]
            raise ValueError(
                f'DNC step {step} invalid response: {e}; response={excerpt!r}'
            ) from e

        # Apply action
        origin_r, origin_c = r, c
        nr, nc = r, c
        if action == 'UP':    nr = r - 1
        elif action == 'DOWN':nr = r + 1
        elif action == 'LEFT':nc = c - 1
        elif action == 'RIGHT':nc = c + 1

        collision = False
        if nr < 0 or nr >= rows or nc < 0 or nc >= cols or (nr, nc) in walls:
            collision = True
            nr, nc = r, c  # bounce back

        if not collision:
            r, c = nr, nc

        # Replay the paper's frozen horizontal-wind entry for the post-action
        # row and this zero-based step. This table is part of the locked scene.
        actual_drift = 0
        nudge = 0
        row_drift = drift_table[r][step - 1]
        if row_drift.get('applies', False):
            gust_size = int(row_drift.get('gust_size', row_drift.get('gustSize', 1)))
            nudge_sign = int(row_drift.get('nudge_sign', row_drift.get('nudgeSign', 1)))
            nudge = nudge_sign * gust_size
            drift_dir = 1 if nudge > 0 else -1
            for offset in range(1, abs(nudge) + 1):
                check_c = c + drift_dir * offset
                if check_c < 0 or check_c >= cols or (r, check_c) in walls:
                    collision = True
                    break
                actual_drift = drift_dir * offset
            c += actual_drift

        # The next prompt must compare the pre-action position with the current
        # post-action/post-wind position. Storing the final position here would
        # always produce a zero displacement and a false "blocked" warning.
        prev_r, prev_c, prev_action = origin_r, origin_c, action

        battery = max(
            0,
            battery - preset['drain_step']
            - (preset['drain_coll'] if collision else 0),
        )
        if collision:
            collisions += 1

        step_log.append(dict(step=step, action=action, row=r, col=c,
                              battery=battery, collision=collision,
                              drift_row=row_drift.get('drift', 0),
                              nudge=nudge, actual_drift=actual_drift,
                              belief=current_belief, raw=raw,
                              pre_row=origin_r, pre_col=origin_c))

        if r == goal_r and c == goal_c:
            end_reason = 'goal'
            break
        if battery <= 0:
            end_reason = 'battery'
            break

    # Final survey
    survey_prompt = build_dnc_survey_prompt(step_log, end_reason)
    try:
        survey = call_llm(survey_prompt, system, api_cfg)
        ctx = _validate_dnc_survey_response(survey)
    except (RuntimeError, ValueError) as e:
        raise ValueError(f'DNC invalid post-trial survey: {e}') from e

    # Match the frozen N100 definition exactly: calculate SI on steps whose
    # recorded drift_row is nonzero, falling back to all steps only if none.
    strategy = _compute_dnc_strategy(step_log)

    return dict(
        ctx=ctx, **strategy,
        ordinalization_reference_version=ORDINALIZATION_REFERENCE['version'],
        ordinalization_centers=list(DNC_REFERENCE_CENTERS),
        steps=step_log, end_reason=end_reason, survey_raw=survey,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# BROWSER-FORMAT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _to_browser_raw_trial(exp, result):
    """Convert a trial result dict to the browser downloadAll raw_trial format."""
    if exp == 'FIP':
        # Browser stores the full LLM response object (factual/context/alloc)
        return result['raw']
    elif exp == 'CTD':
        # Browser stores full trial: patient, steps with per-tick raw, finalized ESI
        return dict(
            T=result['T'],
            patient=result.get('patient'),
            cc=result.get('cc'),
            trueDx=result.get('trueDx'),
            steps=result['steps'],          # list of {tick, ctx, raw, [esi]}
            finalized=dict(ESI=result['sb'], forced=False,
                           tick=result['finalize_tick']),
        )
    else:  # DNC — retain the complete action path for protocol auditing
        return dict(
            steps=result['steps'],
            steps_count=len(result['steps']),
            end_reason=result['end_reason'],
            si=result['si'],
            hpr=result['hpr'],
            vcr=result['vcr'],
            si_fraction=result['si_fraction'],
            si_n_steps=result['si_n_steps'],
            si_step_filter=result['si_step_filter'],
            si_out_of_reference_support=result['si_out_of_reference_support'],
            sb=result['sb'],
            context_belief=result['ctx'],
            survey_raw=result['survey_raw'],
            ordinalization_reference_version=result['ordinalization_reference_version'],
            ordinalization_centers=result['ordinalization_centers'],
        )


def _freeze_to_json(exp, frozen):
    """Return a JSON-serialisable copy of a frozen scenario."""
    if exp == 'DNC':
        return dict(
            walls=sorted([list(w) for w in frozen['walls']]),
            drift_table=frozen['drift_table'],
        )
    # FIP and CTD dicts are already plain Python (lists, dicts, primitives)
    return frozen


def _scenario_hash(frozen_json):
    payload = json.dumps(
        frozen_json, sort_keys=True, separators=(',', ':'), ensure_ascii=False
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _atomic_json_dump(payload, path):
    """Durably replace one JSON file so interruption cannot leave it truncated."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _checkpoint_level_state(pairs, raw_results, browser_raw_trials,
                            attempt_audit, label):
    return dict(
        pairs=pairs,
        raw=raw_results,
        browser_raw_trials=browser_raw_trials,
        attempt_audit=attempt_audit,
        label=label,
    )


def _thaw_scenario(exp, frozen_json):
    """Create a runtime copy of a JSON-serialised frozen scenario."""
    frozen = copy.deepcopy(frozen_json)
    if exp == 'DNC':
        frozen['walls'] = {tuple(cell) for cell in frozen['walls']}
    return frozen


def _portable_path(path):
    """Repository-relative path for provenance (file name if outside the repository)."""
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def _load_scenario_bundle(bundle_path, experiments, levels):
    """Load and verify immutable paper scenarios for the requested tasks."""
    path = Path(bundle_path).resolve()
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('schema_version') != 1:
        raise ValueError(f'Unsupported scenario bundle schema: {path}')

    scenarios = {}
    provenance = dict(
        locked=True,
        bundle_path=_portable_path(path),
        bundle_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_model=payload.get('source_model'),
        source_dataset=payload.get('source_dataset'),
        experiments={},
    )
    for exp in experiments:
        entry = payload.get('experiments', {}).get(exp)
        if not entry:
            raise ValueError(f'{exp} is missing from scenario bundle {path}')
        scenarios[exp] = {}
        level_hashes = {}
        for level in levels:
            item = entry.get('levels', {}).get(str(level))
            if not item:
                raise ValueError(f'{exp} level {level} is missing from {path}')
            scenario_json = item['frozen_scenario']
            observed = _scenario_hash(scenario_json)
            expected = item.get('sha256')
            if observed != expected:
                raise ValueError(
                    f'{exp} level {level} scenario hash mismatch: '
                    f'expected {expected}, observed {observed}'
                )
            scenarios[exp][level] = _thaw_scenario(exp, scenario_json)
            level_hashes[str(level)] = observed
        provenance['experiments'][exp] = dict(
            source_file=entry.get('source_file'),
            source_file_sha256=entry.get('source_file_sha256'),
            level_sha256=level_hashes,
        )
    return scenarios, provenance


# ═══════════════════════════════════════════════════════════════════════════════
# IC ANALYSIS  (Props 1–3)
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_ic(all_level_data):
    """
    Compute IC metrics across all collected conditions.

    Two-property consistency framework
    ─────────────────────────────────────────────────────────────────────────────
    Prop 1 — Contextual-belief reliability
        Pass per condition: RSD(B_C) = SD(B_C) / mean(B_C) × 100 ≤ 20%

    Prop 2 — Risk-decision reliability
        Pass per condition: dominant-class proportion ≥ 0.80

    Prop 3 — Cross-condition monotonicity (supplementary validity check only)
        R_D modes should change monotonically as B_C mean increases across conditions.
        Does NOT contribute to ic_pass — used only for quality flagging.
    ─────────────────────────────────────────────────────────────────────────────
    """
    condition_stats = {}
    all_ctx, all_sb = [], []

    for level in sorted(all_level_data):
        pairs = [(c, s) for c, s in all_level_data[level]['pairs']
                 if c is not None and s is not None]
        if not pairs:
            continue
        ctxs  = [p[0] for p in pairs]
        sbs   = [p[1] for p in pairs]
        n_l   = len(pairs)
        cnt   = Counter(sbs)
        mode_sb, mode_n = cnt.most_common(1)[0]

        ctx_mean = np.mean(ctxs)
        ctx_sd   = np.std(ctxs)
        ctx_rsd  = ctx_sd / ctx_mean * 100 if ctx_mean > 0 else 0.0
        ci_half  = 1.96 * ctx_sd / math.sqrt(n_l) if n_l > 1 else ctx_sd
        band5    = round(sum(1 for c in ctxs if abs(c - ctx_mean) <= 5) / n_l, 3)

        median_sb = float(np.median(sbs))
        pm1_prop  = round(sum(1 for s in sbs if abs(s - median_sb) <= 1) / n_l, 3)
        entropy   = _normalized_entropy(sbs)

        condition_stats[level] = dict(
            n=n_l, label=all_level_data[level].get('label', ''),
            ctx_mean=round(ctx_mean, 1), ctx_sd=round(ctx_sd, 1),
            ctx_min=round(min(ctxs), 1), ctx_max=round(max(ctxs), 1),
            ctx_unique=len(set(ctxs)), sb_unique=len(set(sbs)),
            ctx_rsd=ctx_rsd,
            ctx_ci95=(round(ctx_mean - ci_half, 1), round(ctx_mean + ci_half, 1)),
            ctx_band5_prop=band5,
            sb_mode=mode_sb,    sb_mean=round(np.mean(sbs), 2),
            sb_median=median_sb,
            sb_mode_freq=round(mode_n / n_l, 3),
            sb_entropy=entropy,
            sb_pm1_prop=pm1_prop,
            sb_dist={k: round(v / n_l, 3) for k, v in sorted(cnt.items())},
        )
        all_ctx.extend(ctxs)
        all_sb.extend(sbs)

    if not all_ctx:
        return dict(error='No valid pairs collected')

    # ── Prop 1: Perceptual stability ──────────────────────────────────────────
    p1 = {l: bool(s['ctx_mean'] > 0 and s['ctx_rsd'] <= 20)
          for l, s in condition_stats.items()}

    # ── Prop 2: Behavioral stability ─────────────────────────────────────────
    p2 = {l: bool(s['sb_mode_freq'] >= 0.80)
          for l, s in condition_stats.items()}

    # ── Prop 3: Cross-condition monotonicity (validity check only) ───────────
    ordered = sorted(condition_stats, key=lambda l: condition_stats[l]['ctx_mean'])
    sb_modes_ordered = [condition_stats[l]['sb_mode'] for l in ordered]
    ctx_vals = [condition_stats[l]['ctx_mean'] for l in ordered]
    corr = float(np.corrcoef(ctx_vals, sb_modes_ordered)[0, 1]) \
            if len(set(ctx_vals)) > 1 and len(set(sb_modes_ordered)) > 1 else 0.0
    trend_sign = 1 if corr >= 0 else -1
    mono_violations = 0
    for i in range(len(sb_modes_ordered) - 1):
        diff = sb_modes_ordered[i + 1] - sb_modes_ordered[i]
        if trend_sign == 1 and diff < 0:
            mono_violations += 1
        elif trend_sign == -1 and diff > 0:
            mono_violations += 1

    ctx_range = (round(min(all_ctx), 1), round(max(all_ctx), 1))

    # Build simplified level_stats (only essential stats, no derived fields)
    level_stats = {
        l: dict(
            n             = condition_stats[l]['n'],
            label         = condition_stats[l].get('label', ''),
            ctx_mean      = condition_stats[l]['ctx_mean'],
            ctx_sd        = condition_stats[l]['ctx_sd'],
            ctx_min       = condition_stats[l]['ctx_min'],
            ctx_max       = condition_stats[l]['ctx_max'],
            ctx_unique    = condition_stats[l]['ctx_unique'],
            ctx_rsd       = condition_stats[l]['ctx_rsd'],
            ctx_band5_prop= condition_stats[l]['ctx_band5_prop'],
            sb_mode       = condition_stats[l]['sb_mode'],
            sb_mean       = condition_stats[l]['sb_mean'],
            sb_mode_freq  = condition_stats[l]['sb_mode_freq'],
            sb_entropy    = condition_stats[l]['sb_entropy'],
            sb_pm1_prop   = condition_stats[l]['sb_pm1_prop'],
            prop1_pass    = p1[l],
            prop2_pass    = p2[l],
        )
        for l in condition_stats
    }

    return dict(
        n_total     = len(all_ctx),
        ctx_range   = ctx_range,
        prop1_pass  = int(sum(p1.values())),
        prop2_pass  = int(sum(p2.values())),
        prop3       = dict(corr=round(corr, 3),
                           trend_direction='positive' if corr >= 0 else 'negative',
                           mono_violations=mono_violations, pass_=mono_violations == 0,
                           ordered_levels=ordered,
                           sb_by_ctx=[(condition_stats[l]['ctx_mean'],
                                       condition_stats[l]['sb_mode']) for l in ordered]),
        # Prop 3 is a supplementary validity descriptor and must not gate the
        # stated two-property IC pass criterion.
        ic_pass     = bool(sum(p1.values()) >= 2 and sum(p2.values()) >= 2),
        level_stats = level_stats,
    )


def print_ic_report(metrics, exp_name):
    SEP = '=' * 72
    ok  = lambda b: '[OK]' if b else '[!!]'
    print(f'\n{SEP}')
    print(f'  IC REPORT -- {exp_name.upper()}')
    print(SEP)
    if 'error' in metrics:
        print(f'  ERROR: {metrics["error"]}')
        return

    print(f'  Total pairs : {metrics["n_total"]}')
    print(f'  B_C range   : {metrics["ctx_range"][0]} - {metrics["ctx_range"][1]}')

    # Prop 3: cross-level validity
    p3 = metrics['prop3']
    sb_seq = ' -> '.join(f'B_C={c:.0f}/R_D={s}' for c, s in p3['sb_by_ctx'])
    mono_ok = ok(p3['pass_']) + ('' if p3['pass_'] else f' ({p3["mono_violations"]} violation(s))')
    print(f'  Prop3 Mono  : {mono_ok}  corr={p3["corr"]:+.3f}'
          f'  ({p3["trend_direction"]} trend)  [validity only]')
    print(f'    Sequence  : {sb_seq}')

    # Per-level table
    level_stats = metrics.get('level_stats', {})
    print()
    print(f'  {"Level":<5} {"n":>3}  '
          f'{"B_C mean":>8} {"RSD%":>6}  P1  '
          f'{"R_D mode":>8} {"dom%":>5} {"H":>5}  P2')
    print(f'  {"-"*60}')
    N = len(level_stats)
    for l, s in sorted(level_stats.items()):
        p1ok = ok(s['prop1_pass'])
        p2ok = ok(s['prop2_pass'])
        print(f'  L{l:<4} {s["n"]:>3}  '
              f'{s["ctx_mean"]:>8.1f} {s["ctx_rsd"]:>6.1f}  {p1ok}  '
              f'  R_D={s["sb_mode"]} {s["sb_mode_freq"]:>5.0%}'
              f' {s["sb_entropy"]:>5.3f}  {p2ok}')

    status = 'PASS' if metrics['ic_pass'] else 'FAIL'
    print(f'\n  Overall IC: [{status}]  '
          f'Prop1: {metrics["prop1_pass"]}/{N}  '
          f'Prop2: {metrics["prop2_pass"]}/{N}  '
          f'Mono: {ok(p3["pass_"])}')
    print('  Thresholds: P1 = RSD(B_C)<=20%  |  P2 = dominant R_D>=80%')
    print(f'{SEP}\n')

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

# ── Manipulation-check thresholds (stimulus validity only, not IC outcomes) ──
# Each condition is designed to represent a distinct region of the danger scale.
# After collecting data we verify that the frozen scenario actually induced B_C
# responses in the intended region (a standard manipulation check).  If a level
# deviates beyond MANIP_CHECK_TOLERANCE its stimulus is regenerated.
# Crucially, this check is evaluated independently of Prop 1/2: re-runs are
# never triggered by IC metric outcomes, only by stimulus invalidity.
MANIP_CHECK_TARGETS   = {1: 15, 2: 50, 3: 85}  # L1=safe, L2=neutral, L3=dangerous
MANIP_CHECK_TOLERANCE = 30                        # max |B_C_mean − target| to accept


def _generate_frozen(exp, level, preset):
    """Generate a frozen scenario/environment for one level."""
    if exp == 'FIP':
        frozen = generate_frozen_fip_market(preset)
        info = (f'turb={frozen["states"].count("turb")}/{len(frozen["states"])} '
                f'({frozen["ctx_true_pct"]}% turb), trends={frozen["gt_trend"]}')
    elif exp == 'CTD':
        frozen = generate_frozen_ctd_scenario(preset)
        info = (f'age={frozen["patient"]["age"]}, '
                f'comorbid={frozen["patient"]["comorbid"]}, '
                f'dx={frozen["trueDx"]}, T={frozen["T"]}')
    else:  # DNC
        frozen = generate_frozen_dnc_env(level, preset)
        info = f'walls={len(frozen["walls"])}, drift_table={GRID_ROWS}×{preset["max_steps"]+10}'
    return frozen, info


def _run_level(exp, level, preset, frozen, api_cfg, n_trials,
               max_extra_attempts=0, max_consecutive_failures=0,
               initial_state=None, checkpoint_every=10,
               checkpoint_callback=None):
    """Run N trials for one condition using the given frozen scenario.
    When max_extra_attempts > 0, invalid responses are replaced by additional
    attempts on that same frozen scenario until N valid trials are collected.
    Returns (pairs, raw_results, browser_raw_trials, attempt_audit).
    """
    initial_state = initial_state or {}
    pairs = copy.deepcopy(initial_state.get('pairs', []))
    raw_results = copy.deepcopy(initial_state.get('raw', []))
    browser_raw_trials = copy.deepcopy(
        initial_state.get('browser_raw_trials', [])
    )
    attempt_audit = copy.deepcopy(initial_state.get('attempt_audit', []))
    attempts = len(pairs)
    valid_count = len(raw_results)
    consecutive_failures = 0
    attempt_limit = n_trials + max_extra_attempts
    if attempts or valid_count:
        print(f'  Resuming checkpoint: {valid_count}/{n_trials} valid, '
              f'{attempts}/{attempt_limit} attempts already used.')

    def emit_checkpoint():
        if checkpoint_callback is None:
            return
        checkpoint_callback(_checkpoint_level_state(
            pairs, raw_results, browser_raw_trials, attempt_audit,
            preset['label'],
        ))

    while valid_count < n_trials and attempts < attempt_limit:
        attempts += 1
        checkpoint_due = False
        print(f'  Trial {valid_count+1}/{n_trials} (attempt {attempts}/{attempt_limit})…',
              end=' ', flush=True)
        try:
            if exp == 'FIP':
                result = run_fip_trial(frozen, preset, api_cfg)
            elif exp == 'CTD':
                result = run_ctd_trial(frozen, preset, api_cfg)
            else:  # DNC
                result = run_dnc_trial(frozen, level, preset, api_cfg)

            if result:
                pairs.append((result['ctx'], result['sb']))
                raw_results.append(result)
                browser_raw_trials.append(_to_browser_raw_trial(exp, result))
                print(f'B_C={result["ctx"]:.0f} R_D={result["sb"]}')
                valid_count += 1
                consecutive_failures = 0
                attempt_audit.append(dict(
                    attempt=attempts, status='valid', valid_index=valid_count,
                ))
                checkpoint_due = (
                    checkpoint_every > 0
                    and valid_count % checkpoint_every == 0
                )
            else:
                print('skip (bad response)')
                pairs.append((None, None))
                consecutive_failures += 1
                attempt_audit.append(dict(
                    attempt=attempts, status='invalid',
                    reason='trial runner returned no valid result',
                ))
        except Exception as e:
            print(f'ERROR: {e}')
            pairs.append((None, None))
            consecutive_failures += 1
            attempt_audit.append(dict(
                attempt=attempts, status='invalid',
                error_type=type(e).__name__, reason=str(e),
            ))
        # Checkpoint I/O is deliberately outside the trial's exception handler:
        # a disk failure must not relabel an otherwise valid model response as
        # an invalid trial or corrupt the attempt accounting.
        if checkpoint_due:
            emit_checkpoint()
        if (max_consecutive_failures > 0
                and consecutive_failures >= max_consecutive_failures):
            # Persist the tail (including invalid-attempt audit entries) before
            # deliberately aborting the level.
            emit_checkpoint()
            raise RuntimeError(
                f'Aborting {exp} L{level} after {consecutive_failures} '
                'consecutive invalid trials; inspect the last API response.'
            )
    # Preserve a non-multiple-of-ten tail and the complete invalid-attempt
    # audit before analysis/final publication writes begin.
    emit_checkpoint()
    if valid_count < n_trials:
        print(f'  WARNING: collected {valid_count}/{n_trials} valid trials after '
              f'{attempts} attempts.')
    return pairs, raw_results, browser_raw_trials, attempt_audit


def _generation_audit(exp, pairs, raw_results):
    """Describe degeneracy and scale risks without changing observed outcomes."""
    valid_pairs = [(float(c), int(s)) for c, s in pairs
                   if c is not None and s is not None]
    bc_values = [c for c, _ in valid_pairs]
    rd_values = [s for _, s in valid_pairs]
    raw_hashes = {
        _scenario_hash(result)
        for result in raw_results
    }
    audit = dict(
        valid_trials=len(valid_pairs),
        unique_bc=len(set(bc_values)),
        unique_rd=len(set(rd_values)),
        unique_raw_trials=len(raw_hashes),
        constant_bc=(len(set(bc_values)) == 1 if bc_values else None),
        constant_rd=(len(set(rd_values)) == 1 if rd_values else None),
    )
    if exp == 'CTD' and valid_pairs:
        scale_coded = [
            c.is_integer() and 0 <= c <= 5 for c in bc_values
        ]
        audit.update(
            bc_integer_0_to_5_rate=sum(scale_coded) / len(scale_coded),
            bc_equals_rd_rate=sum(c == s for c, s in valid_pairs) / len(valid_pairs),
            finalize_tick1_rate=(
                sum(result.get('finalize_tick') == 1 for result in raw_results)
                / len(raw_results) if raw_results else None
            ),
            premature_finalization_attempt_rate=(
                sum(
                    any('premature_finalize_ESI' in step
                        for step in result.get('steps', []))
                    for result in raw_results
                ) / len(raw_results) if raw_results else None
            ),
        )
    elif exp == 'DNC' and raw_results:
        si_values = [float(result['si']) for result in raw_results]
        audit.update(
            unique_si=len(set(si_values)),
            si_mean=float(np.mean(si_values)),
            si_sd=float(np.std(si_values)),
            si_min=min(si_values),
            si_max=max(si_values),
            si_out_of_reference_support=sum(
                bool(result['si_out_of_reference_support'])
                for result in raw_results
            ),
        )
    elif exp == 'FIP' and raw_results:
        totals = [float(result['allocation_total']) for result in raw_results]
        audit.update(
            allocation_sum_100_rate=sum(
                math.isclose(total, 100.0, rel_tol=0.0, abs_tol=1e-6)
                for total in totals
            ) / len(totals),
            allocation_total_min=min(totals),
            allocation_total_max=max(totals),
        )
    return audit


def _print_generation_audit(exp, all_level_data, levels):
    print('  Generation audit:')
    for level in sorted(levels):
        item = all_level_data[level]
        audit = _generation_audit(exp, item['pairs'], item['raw'])
        if not audit['valid_trials']:
            print(f'    L{level}: no valid trials')
            continue
        line = (
            f'    L{level}: unique B_C={audit["unique_bc"]}, '
            f'unique R_D={audit["unique_rd"]}, '
            f'unique raw trials={audit["unique_raw_trials"]}'
        )
        if exp == 'CTD':
            line += (
                f', B_C integer 0-5={audit["bc_integer_0_to_5_rate"]:.0%}, '
                f'B_C=R_D={audit["bc_equals_rd_rate"]:.0%}, '
                f'finalize@tick1={audit["finalize_tick1_rate"]:.0%}, '
                f'premature-attempt trials='
                f'{audit["premature_finalization_attempt_rate"]:.0%}'
            )
        elif exp == 'DNC':
            line += (
                f', unique SI={audit["unique_si"]}, '
                f'SI mean={audit["si_mean"]:.3f}, '
                f'range={audit["si_min"]:.3f}..{audit["si_max"]:.3f}'
            )
        elif exp == 'FIP':
            line += (
                f', allocation sum=100 rate='
                f'{audit["allocation_sum_100_rate"]:.0%}'
            )
        print(line)
    print()


def _level_summary(all_level_data, levels):
    """
    Return per-level stats: {level: {ctx_mean, sb_mode, sb_mean, n}}.
    Only includes levels with at least one valid pair.
    """
    stats = {}
    for level in levels:
        valid = [(c, s) for c, s in all_level_data[level]['pairs']
                 if c is not None and s is not None]
        if not valid:
            continue
        ctxs = [p[0] for p in valid]
        sbs  = [p[1] for p in valid]
        cnt  = Counter(sbs)
        stats[level] = dict(
            ctx_mean=np.mean(ctxs),
            sb_mode=cnt.most_common(1)[0][0],
            sb_mean=np.mean(sbs),
            n=len(valid),
        )
    return stats


def _invalid_conditions(all_level_data, levels):
    """
    Manipulation check: identify levels whose frozen scenario failed to induce
    B_C responses in the intended danger region.

    A level is invalid if its observed B_C mean deviates by more than
    MANIP_CHECK_TOLERANCE from MANIP_CHECK_TARGETS[level].  This is a standard
    stimulus-validity check (analogous to a manipulation check in experimental
    psychology) and is the sole criterion for triggering a re-run.

    Prop 1 / Prop 2 outcomes are intentionally excluded from this check:
    IC metrics are measured outcomes, not grounds for regenerating stimuli.

    Returns a list of (level, reason_str), sorted by deviation magnitude.
    """
    stats = _level_summary(all_level_data, levels)
    problems = []
    for level in sorted(levels):
        s = stats.get(level)
        if s is None:
            continue
        target = MANIP_CHECK_TARGETS.get(level, 50)
        dev    = s['ctx_mean'] - target
        if abs(dev) > MANIP_CHECK_TOLERANCE:
            direction = 'too high' if dev > 0 else 'too low'
            problems.append((level,
                f'B_C mean={s["ctx_mean"]:.1f} is {direction} '
                f'(target≈{target} ± {MANIP_CHECK_TOLERANCE}); '
                f'stimulus does not represent the intended danger level'))
    problems.sort(key=lambda kv: abs(_level_summary(all_level_data, levels)
                                     .get(kv[0], {}).get('ctx_mean', 0)
                                     - MANIP_CHECK_TARGETS.get(kv[0], 50)),
                  reverse=True)
    return problems


def run_ic_experiment(exp, api_cfg, n_trials=10, levels=None,
                      max_reruns=3, out_dir=DEFAULT_OUT, seed=None,
                      max_extra_attempts=0, frozen_scenarios=None,
                      scenario_provenance=None, organized_output=False,
                      run_id=None, output_model_name=None,
                      max_consecutive_failures=0, checkpoint_every=10,
                      resume=False):
    """
    Full Phase-1 (reliability) collection for one task.

    Step 1 — Run all levels once with a locked or freshly generated scenario.
    Step 2 — Manipulation check: if any level's B_C mean falls outside its
              intended danger region (MANIP_CHECK_TARGETS ± MANIP_CHECK_TOLERANCE),
              that level's stimulus is regenerated and re-run (up to max_reruns
              rounds).  Re-runs are triggered solely by stimulus invalidity —
              never by whether Prop 1 / Prop 2 pass.  IC metrics are always
              reported as final observed outcomes.

    Reproducibility: locked scenarios are verified by SHA-256 and stored in the
    output metadata. The random seed is also fixed and recorded.
    """
    # ── Fix random seeds for reproducibility ─────────────────────────────────
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    random.seed(seed)
    np.random.seed(seed)

    PRESETS = dict(FIP=FIP_PRESETS, CTD=CTD_PRESETS, DNC=DNC_PRESETS)[exp]
    if levels is None:
        levels = sorted(PRESETS)

    output_model_name = output_model_name or _model_output_name(api_cfg['model'])
    if (not output_model_name or output_model_name.strip('_') != output_model_name
            or any(not (ch.isalnum() or ch == '_') for ch in output_model_name)):
        raise ValueError('output_model_name must contain only letters, numbers, and underscores')
    if organized_output:
        if not run_id:
            raise ValueError('run_id is required with organized_output')
        if (run_id.strip('_-') != run_id
                or any(not (ch.isalnum() or ch in '_-') for ch in run_id)):
            raise ValueError('run_id must contain only letters, numbers, underscores, and hyphens')
        out_path = Path(out_dir) / exp / output_model_name / run_id
    else:
        out_path = Path(out_dir) / exp
    out_path.mkdir(parents=True, exist_ok=True)

    print(f'  Scenario seed: {seed}')
    print('  LLM sampling : provider-managed stochastic sampling (no fixed request seed)')
    print(f'  Model       : {api_cfg["model"]}')
    print(f'  Protocol    : {PROTOCOL_NAME}')
    if exp in ('DNC', 'FIP'):
        print(f'  Ordinal map : {ORDINALIZATION_REFERENCE["version"]} frozen centers')
    if frozen_scenarios is not None:
        print('  Scenarios   : LOCKED (paper Qwen3 Max source bundle)')

    all_level_data   = {}
    rerun_count      = {l: 0 for l in levels}
    frozen_per_level = {}
    browser_trials   = {}
    checkpoint_path = out_path / '_checkpoints' / f'{exp}_checkpoint.json'
    inference_fingerprint = dict(
        model=api_cfg['model'],
        provider=api_cfg.get('provider'),
        base_url=api_cfg.get('base_url'),
        api_mode=api_cfg.get('api_mode'),
        temperature=api_cfg.get('temperature'),
        top_p=api_cfg.get('top_p'),
        max_tokens=api_cfg.get('max_tokens'),
        timeout=api_cfg.get('timeout'),
        max_retries=api_cfg.get('max_retries'),
        invalidate_on_api_error=bool(api_cfg.get('invalidate_on_api_error')),
        completion_assistant_cue=bool(api_cfg.get('completion_assistant_cue')),
    )
    scenario_hashes = (
        {
            str(level): _scenario_hash(
                _freeze_to_json(exp, frozen_scenarios[level])
            )
            for level in levels
        }
        if frozen_scenarios is not None else None
    )
    resume_states = {}
    if resume:
        if frozen_scenarios is None:
            raise ValueError('Resume requires a locked scenario bundle')
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f'Cannot resume: checkpoint does not exist: {checkpoint_path}'
            )
        checkpoint = json.loads(checkpoint_path.read_text(encoding='utf-8'))
        expected_identity = dict(
            schema_version=1,
            experiment=exp,
            protocol=PROTOCOL_NAME,
            model=api_cfg['model'],
            run_id=run_id,
            scenario_seed=seed,
            n_trials_per_level=n_trials,
            levels=list(levels),
            max_extra_attempts=max_extra_attempts,
            max_consecutive_failures=max_consecutive_failures,
            inference_fingerprint=inference_fingerprint,
            scenario_hashes=scenario_hashes,
        )
        observed_identity = {
            key: checkpoint.get(key) for key in expected_identity
        }
        if observed_identity != expected_identity:
            raise ValueError(
                'Checkpoint identity does not match this run configuration; '
                'refusing to mix protocols, models, scenarios, or attempt budgets.'
            )
        resume_states = {
            int(level): state
            for level, state in checkpoint.get('level_states', {}).items()
        }
        print(f'  Resume      : {checkpoint_path}')

    def save_checkpoint(status='collecting', current_level=None,
                        current_state=None):
        level_states = {}
        for saved_level, item in all_level_data.items():
            level_states[str(saved_level)] = _checkpoint_level_state(
                item['pairs'], item['raw'], browser_trials[saved_level],
                item.get('attempt_audit', []), item.get('label', ''),
            )
        if current_level is not None and current_state is not None:
            level_states[str(current_level)] = current_state
        payload = dict(
            schema_version=1,
            status=status,
            saved_at=datetime.now().isoformat(timespec='seconds'),
            experiment=exp,
            protocol=PROTOCOL_NAME,
            model=api_cfg['model'],
            run_id=run_id,
            scenario_seed=seed,
            n_trials_per_level=n_trials,
            levels=list(levels),
            max_extra_attempts=max_extra_attempts,
            max_consecutive_failures=max_consecutive_failures,
            inference_fingerprint=inference_fingerprint,
            scenario_hashes=scenario_hashes,
            level_states=level_states,
        )
        _atomic_json_dump(payload, checkpoint_path)
        valid_saved = sum(len(state.get('raw', [])) for state in level_states.values())
        print(f'  Checkpoint  : {valid_saved} valid records -> {checkpoint_path}')

    # ── Step 1: run every level once ─────────────────────────────────────────
    print(f'\n{"─"*60}')
    print(f'  STEP 1 — running all {len(levels)} levels')
    print(f'{"─"*60}')

    for level in levels:
        preset = PRESETS[level]
        print(f'\n[{exp}] L{level}: {preset["label"]}')
        if frozen_scenarios is not None:
            frozen = copy.deepcopy(frozen_scenarios[level])
            digest = _scenario_hash(_freeze_to_json(exp, frozen))
            info = f'LOCKED paper scenario sha256={digest[:12]}'
        else:
            frozen, info = _generate_frozen(exp, level, preset)
        print(f'  Frozen scenario: {info}')
        pairs, raw, br_trials, attempt_audit = _run_level(
            exp, level, preset, frozen, api_cfg, n_trials,
            max_extra_attempts=max_extra_attempts,
            max_consecutive_failures=max_consecutive_failures,
            initial_state=resume_states.get(level),
            checkpoint_every=checkpoint_every,
            checkpoint_callback=(
                lambda state, current_level=level: save_checkpoint(
                    current_level=current_level, current_state=state
                )
            ),
        )
        all_level_data[level]   = dict(
            pairs=pairs, raw=raw, attempt_audit=attempt_audit,
            label=preset['label'],
        )
        frozen_per_level[level] = frozen
        browser_trials[level]   = br_trials
        rerun_count[level]      = 1

    # For locked paper scenarios, manipulation checks are diagnostic only.
    # The stimuli must never be regenerated based on model responses.
    if frozen_scenarios is not None:
        problems = _invalid_conditions(all_level_data, levels)
        if problems:
            print(f'\n  Manipulation check flagged {len(problems)} condition(s). '
                  'Scenarios are locked; reporting results without regeneration.')
        else:
            print(f'\n  All locked conditions passed manipulation check '
                  f'(|B_C_mean - target| <= {MANIP_CHECK_TOLERANCE}).')
    else:
        # Unlocked mode: regenerate stimuli that fail the manipulation check.
        for rerun_round in range(max_reruns):
            problems = _invalid_conditions(all_level_data, levels)
            if not problems:
                print(f'\n  All conditions passed manipulation check '
                      f'(|B_C_mean - target| <= {MANIP_CHECK_TOLERANCE}).')
                break

            print(f'\n{"-"*60}')
            print(f'  MANIPULATION CHECK round {rerun_round + 1} - '
                  f'regenerating {len(problems)} invalid stimulus/stimuli')
            print(f'{"-"*60}')
            for level, reason in problems:
                print(f'\n  Re-running L{level} (manipulation check failed): {reason}')
                preset = PRESETS[level]
                frozen, info = _generate_frozen(exp, level, preset)
                print(f'  New frozen scenario: {info}')
                pairs, raw, br_trials, attempt_audit = _run_level(
                    exp, level, preset, frozen, api_cfg, n_trials,
                    max_extra_attempts=max_extra_attempts,
                    max_consecutive_failures=max_consecutive_failures,
                )
                all_level_data[level] = dict(
                    pairs=pairs, raw=raw, attempt_audit=attempt_audit,
                    label=preset['label']
                )
                frozen_per_level[level] = frozen
                browser_trials[level] = br_trials
                rerun_count[level] += 1
        else:
            remaining = _invalid_conditions(all_level_data, levels)
            if remaining:
                print(f'\n  WARNING: reached max_reruns={max_reruns}; '
                      f'{len(remaining)} level(s) still outside manipulation-check range. '
                      f'IC metrics are reported as observed.')

    # ── Final IC analysis — always computed; never used to gate re-runs ───────
    metrics = analyze_ic(all_level_data)
    print_ic_report(metrics, f'{exp} (final)')
    _print_generation_audit(exp, all_level_data, levels)

    # ── Manipulation-check summary ────────────────────────────────────────────
    manip_check_summary = {}
    for l in levels:
        valid = [(c, s) for c, s in all_level_data[l]['pairs']
                 if c is not None and s is not None]
        if valid:
            ctx_mean = float(np.mean([p[0] for p in valid]))
            target   = MANIP_CHECK_TARGETS.get(l, 50)
            manip_check_summary[l] = dict(
                bc_mean           = round(ctx_mean, 1),
                target            = target,
                deviation         = round(ctx_mean - target, 1),
                passed            = abs(ctx_mean - target) <= MANIP_CHECK_TOLERANCE,
                stimuli_generated = rerun_count[l],
            )

    # ── Save results ──────────────────────────────────────────────────────────
    short_model = _model_short_name(api_cfg['model'])
    total_n     = sum(
        sum(1 for c, s in all_level_data[l]['pairs'] if c is not None and s is not None)
        for l in levels
    )
    base_name = (f'{exp}_{output_model_name}_N{total_n}'
                 if organized_output else f'{exp}_{short_model}_N{total_n}')

    # ── Primary data file: flat array format (matches FrontierModel_reliability_N90) ──
    model_id      = api_cfg['model']
    model_display = MODEL_DISPLAY_NAMES.get(model_id, model_id.split('/')[-1])
    model_key     = short_model   # already computed above via _model_short_name

    records = []
    global_idx = 0
    for l in sorted(levels):
        level_idx = 0
        for c, s in all_level_data[l]['pairs']:
            if c is not None and s is not None:
                record = {
                    'experiment':        exp,
                    'model':             model_display,
                    'model_key':         model_key,
                    'level':             l,
                    'trial_index':       global_idx,
                    'level_trial_index': level_idx,
                    'contextual_belief': float(c),
                    'risk_decision':     int(s),
                }
                records.append(record)
                global_idx += 1
                level_idx  += 1

    data_fname = out_path / f'{base_name}.json'
    _atomic_json_dump(records, data_fname)
    print(f'\nData saved   -> {data_fname}')

    # ── Sidecar provenance file: seed, model, QC details, raw trials ─────────
    configs = []
    for l in sorted(levels):
        valid_pairs = [(c, s) for c, s in all_level_data[l]['pairs']
                       if c is not None and s is not None]
        configs.append(dict(
            preset_level    = l,
            env_config      = PRESETS[l],
            n_trials        = len(valid_pairs),
            attempts        = len(all_level_data[l]['pairs']),
            invalid_trials  = len(all_level_data[l]['pairs']) - len(valid_pairs),
            attempt_audit   = all_level_data[l].get('attempt_audit', []),
            generation_audit= _generation_audit(
                exp, all_level_data[l]['pairs'], all_level_data[l]['raw']
            ),
            frozen_scenario = _freeze_to_json(exp, frozen_per_level[l]),
            raw_trials      = browser_trials[l],
        ))

    meta_payload = dict(
        experiment         = exp,
        protocol           = PROTOCOL_NAME,
        seed               = seed,
        scenario_seed      = seed,
        model              = api_cfg['model'],
        inference_config   = dict(
            provider    = api_cfg.get('provider'),
            api_mode    = api_cfg.get('api_mode'),
            temperature = api_cfg.get('temperature'),
            top_p       = api_cfg.get('top_p'),
            max_tokens  = api_cfg.get('max_tokens'),
            timeout     = api_cfg.get('timeout'),
            max_retries = api_cfg.get('max_retries'),
            sampling_seed = None,
            completion_assistant_cue = bool(api_cfg.get('completion_assistant_cue')),
            completion_json_prefix = (
                CTD_COMPLETION_JSON_PREFIX
                if exp == 'CTD' and api_cfg.get('api_mode') == 'completions'
                else None
            ),
        ),
        n_trials_per_level = n_trials,
        max_extra_attempts = max_extra_attempts,
        max_consecutive_failures = max_consecutive_failures,
        invalidate_on_api_error = bool(api_cfg.get('invalidate_on_api_error')),
        ordinalization_reference = (
            dict(
                path=_portable_path(REFERENCE_MAP_PATH),
                version=ORDINALIZATION_REFERENCE['version'],
                task=ORDINALIZATION_REFERENCE[exp],
            ) if exp in ('DNC', 'FIP') else None
        ),
        scenario_source    = scenario_provenance,
        output_layout      = ('task/model/run' if organized_output else 'task'),
        run_id             = run_id,
        manipulation_check = manip_check_summary,
        rerun_counts       = rerun_count,
        ic_analysis        = metrics,
        configs            = configs,
    )
    meta_fname = out_path / f'{base_name}_meta.json'
    _atomic_json_dump(meta_payload, meta_fname)
    print(f'Meta saved   → {meta_fname}')

    expected_total = len(levels) * n_trials
    save_checkpoint(
        status=('complete' if total_n == expected_total
                else 'attempt_budget_exhausted')
    )

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Phase-1 (frozen-scenario) LLM data collection for DNC / CTD / FIP. '
                    'Keys come from OPENROUTER_API_KEY (or OPENROUTER_KEY) / HF_TOKEN.')
    parser.add_argument('--provider',   default='openrouter',
                        choices=['openrouter', 'huggingface'],
                        help='API provider (default: openrouter)')
    parser.add_argument('--exp',        default='FIP', choices=['FIP','CTD','DNC','all'],
                        help='Experiment to run: FIP, CTD, DNC, or all (default: FIP)')
    parser.add_argument('--n',          type=int, default=10,
                        help='Trials per level (default: 10)')
    parser.add_argument('--levels',     default='1,2,3',
                        help='Comma-separated conditions to run (default: 1,2,3)')
    parser.add_argument('--model',      required=True,
                        help='Exact served model ID, e.g. qwen/qwen3-max or Qwen/Qwen3-30B-A3B-Base')
    parser.add_argument('--base-url',   default='',
                        help='OpenAI-compatible API root (default: OpenRouter, or HF_ENDPOINT_URL for huggingface)')
    parser.add_argument('--api-mode',   default='chat', choices=['chat','completions'],
                        help='Use chat/completions or raw completions (default: chat)')
    parser.add_argument('--temperature',type=float, default=0.7)
    parser.add_argument('--top-p',      type=float, default=1.0)
    parser.add_argument('--max-tokens', type=int,   default=512)
    parser.add_argument('--timeout',    type=int,   default=90)
    parser.add_argument('--api-retries', type=int, default=4,
                        help='Maximum attempts for each individual API request (default: 4)')
    parser.add_argument('--completion-assistant-cue', action='store_true',
                        help='Append an explicit assistant JSON cue in raw-completions mode')
    parser.add_argument('--reruns',     type=int,   default=3,
                        help='Max stimulus-regeneration rounds (manipulation check only, default: 3)')
    parser.add_argument('--max-extra-attempts', type=int, default=0,
                        help='Extra attempts per frozen condition used to replace invalid trials (default: 0)')
    parser.add_argument('--max-consecutive-failures', type=int, default=0,
                        help='Abort a level after this many consecutive invalid trials (0 disables)')
    parser.add_argument('--checkpoint-every', type=int, default=10,
                        help='Atomically checkpoint after this many valid trials (default: 10)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume the exact same run ID from its checkpoint')
    parser.add_argument('--invalidate-on-api-error', action='store_true',
                        help='Mark a multi-step CTD/DNC trial invalid after any API error')
    parser.add_argument('--seed',       type=int,   default=None,
                        help='Local scenario-generation seed; does not seed provider sampling')
    parser.add_argument('--scenario-bundle', default='',
                        help='Locked frozen-scenario bundle; disables manipulation-based regeneration')
    parser.add_argument('--out',        default=str(DEFAULT_OUT),
                        help='Output directory (default: outputs/experiments/)')
    parser.add_argument('--organized-output', action='store_true',
                        help='Write as <out>/<task>/<model>/<run-id>/')
    parser.add_argument('--run-id', default='',
                        help='Run identifier used by --organized-output')
    parser.add_argument('--output-model-name', default='',
                        help='Stable model folder/file component override')
    parser.add_argument('--dry-run',    action='store_true',
                        help='Validate and print the run plan without making API calls')
    args = parser.parse_args()

    if args.organized_output and not args.run_id:
        parser.error('--run-id is required with --organized-output')
    if args.resume and not args.organized_output:
        parser.error('--resume requires --organized-output and the original --run-id')
    if args.checkpoint_every < 1:
        parser.error('--checkpoint-every must be at least 1')

    # Keys are read from the environment only (never from the command line).
    if args.provider == 'huggingface':
        key = os.environ.get('HF_TOKEN', '')
        base_url = args.base_url or os.environ.get('HF_ENDPOINT_URL', '')
        env_name = 'HF_TOKEN'
    else:
        key = os.environ.get('OPENROUTER_API_KEY') or os.environ.get('OPENROUTER_KEY', '')
        base_url = args.base_url or 'https://openrouter.ai/api/v1'
        env_name = 'OPENROUTER_API_KEY'

    if not args.dry_run and not key:
        sys.exit(f'ERROR: set the {env_name} environment variable.')
    if not args.dry_run and not base_url:
        sys.exit('ERROR: Provide --base-url or set HF_ENDPOINT_URL.')

    api_cfg = dict(
        key=key, model=args.model, provider=args.provider,
        temperature=args.temperature, top_p=args.top_p,
        max_tokens=args.max_tokens, timeout=args.timeout,
        base_url=base_url, api_mode=args.api_mode,
        invalidate_on_api_error=args.invalidate_on_api_error,
        max_retries=args.api_retries,
        completion_assistant_cue=args.completion_assistant_cue,
    )
    levels = [int(x) for x in args.levels.split(',')]
    exps   = ['FIP','CTD','DNC'] if args.exp == 'all' else [args.exp]
    locked_scenarios = None
    scenario_provenance = None
    if args.scenario_bundle:
        locked_scenarios, scenario_provenance = _load_scenario_bundle(
            args.scenario_bundle, exps, levels
        )

    if args.dry_run:
        print('DRY RUN — no API requests will be made')
        print(f'  Provider          : {args.provider}')
        print(f'  Model             : {args.model}')
        print(f'  API mode          : {args.api_mode}')
        print(f'  Experiments       : {", ".join(exps)}')
        print(f'  Protocol          : {PROTOCOL_NAME}')
        if 'DNC' in exps:
            print(f'  DNC SI mapping    : {ORDINALIZATION_REFERENCE["version"]} frozen centers')
        print(f'  Conditions        : {levels}')
        print(f'  Trials/condition  : {args.n}')
        print(f'  Target records    : {len(exps) * len(levels) * args.n}')
        print(f'  Scenario seed     : {args.seed}')
        print('  LLM sampling seed : provider-managed (not fixed)')
        if scenario_provenance:
            print(f'  Scenario bundle   : {scenario_provenance["bundle_path"]}')
            print(f'  Bundle SHA-256    : {scenario_provenance["bundle_sha256"]}')
            for exp in exps:
                hashes = scenario_provenance['experiments'][exp]['level_sha256']
                print(f'  {exp} scenario SHA : ' + ', '.join(
                    f'L{level}={digest[:12]}' for level, digest in hashes.items()
                ))
        print(f'  Temperature       : {args.temperature}')
        print(f'  Top-p             : {args.top_p}')
        print(f'  Max output tokens : {args.max_tokens}')
        print(f'  API attempts/call : {args.api_retries}')
        print(f'  Completion cue    : {args.completion_assistant_cue}')
        print(f'  Manip. reruns max : {args.reruns}')
        print(f'  Extra attempts max: {args.max_extra_attempts}')
        print(f'  Consecutive fail max: {args.max_consecutive_failures}')
        print(f'  Checkpoint interval: {args.checkpoint_every} valid trials')
        print(f'  Resume checkpoint : {args.resume}')
        print(f'  Invalidate trial after API error: {args.invalidate_on_api_error}')
        print(f'  Output            : {Path(args.out).resolve()}')
        if args.organized_output:
            print(f'  Output layout     : task/model/{args.run_id}')
            print(f'  Output model name : {args.output_model_name or _model_output_name(args.model)}')
        return

    for exp in exps:
        run_ic_experiment(
            exp, api_cfg, n_trials=args.n, levels=levels,
            max_reruns=args.reruns, out_dir=args.out, seed=args.seed,
            max_extra_attempts=args.max_extra_attempts,
            frozen_scenarios=(locked_scenarios or {}).get(exp),
            scenario_provenance=scenario_provenance,
            organized_output=args.organized_output,
            run_id=args.run_id,
            output_model_name=(args.output_model_name or None),
            max_consecutive_failures=args.max_consecutive_failures,
            checkpoint_every=args.checkpoint_every,
            resume=args.resume,
        )

if __name__ == '__main__':
    main()
