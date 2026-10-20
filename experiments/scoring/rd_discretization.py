"""Score Phase-2 (attitude-estimation) trials: contextual belief B_C and ordinal risk decision R_D.

Works on the dataset's *_main_N100 files and on raw downloads from the browser interfaces
(experiments/web/llm), which share the per-trial layout.

  Task  B_C                                  raw decision                           R_D
  DNC   post-trial survey belief             Strategy Index SI on drift steps       nearest frozen SI center
  FIP   report.contextual.risk               high-risk share H/(L+M+H)              nearest frozen share center
  CTD   last BC_updates[].ctx of the trial   finalized ESI                          ESI (native 1..5)

SI = logit(VCR / (VCR + HPR + 0.001)) with the fraction clipped to [0.001, 0.999]; VCR / HPR
are the shares of UP|DOWN / RIGHT actions on steps whose latent drift row is non-zero (all
steps if there are none). In the interior this equals log(VCR/HPR); the clipping bounds SI
to +-6.9068. Larger SI = more vertical correction = more cautious (R_D = 1).

Benchmark v1 freezes the five centers per task (experiments/config/frozen_ordinalization_map.json);
new observations are assigned to the nearest center (ties to the smaller label) without refitting,
and values outside the reference range are flagged. The centers were obtained once by pooled
one-dimensional K-means (k=5, n_init=10, random_state=42) ordered by centroid; `--refit` shows
that procedure on the files given (refitting on a different pool gives different centers).

Usage:
    python experiments/scoring/rd_discretization.py data/hf/FrontierModel_main_N100/*/*.json
    python experiments/scoring/rd_discretization.py --refit data/hf/FrontierModel_main_N100/DNC/*.json
"""
import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from riskattitude.paths import OUTPUT_DIR, REPO_ROOT  # noqa: E402

MAP_PATH = REPO_ROOT / 'experiments' / 'config' / 'frozen_ordinalization_map.json'
REFERENCE = json.loads(MAP_PATH.read_text(encoding='utf-8'))
CENTERS = {task: [float(c) for c in REFERENCE[task]['centers_in_R_D_order']] for task in ('DNC', 'FIP')}
SUPPORT = {'DNC': (REFERENCE['DNC']['observed_reference_range']['minimum'],
                   REFERENCE['DNC']['observed_reference_range']['maximum']),
           'FIP': (0.0, 0.8)}   # observed v1 range of H/(L+M+H)
DNC_EPS = 0.001


# ── per-trial extraction ────────────────────────────────────────────────────

def dnc_strategy_index(steps) -> float:
    drift = [s for s in steps if s.get('drift_row', 0) != 0] or steps
    vcr = sum(str(s.get('action', '')).upper() in ('UP', 'DOWN') for s in drift) / len(drift)
    hpr = sum(str(s.get('action', '')).upper() == 'RIGHT' for s in drift) / len(drift)
    fraction = min(max(vcr / (vcr + hpr + DNC_EPS), DNC_EPS), 1 - DNC_EPS)
    return math.log(fraction / (1 - fraction))


def extract(trial: dict, task: str):
    """Return (B_C, raw decision) or None if the trial lacks a usable belief/decision."""
    if task == 'DNC':
        steps = trial.get('steps') or []
        if not steps:
            return None
        bc = trial.get('context_belief', trial.get('contextual_belief', steps[-1].get('belief')))
        return None if bc is None else (float(bc), dnc_strategy_index(steps))
    if task == 'FIP':
        report = trial.get('report') or {}
        bc = (report.get('contextual') or {}).get('risk')
        alloc = report.get('alloc') or {}
        weights = [float(alloc.get(k) or 0) for k in ('L', 'M', 'H')]
        if bc is None or sum(weights) == 0:
            return None
        return float(bc), weights[2] / sum(weights)
    bc = None                                   # CTD
    for step in reversed(trial.get('steps') or []):
        if step.get('BC_updates'):
            bc = float(step['BC_updates'][-1]['ctx'])
            break
    esi = (trial.get('finalized') or {}).get('ESI')
    match = re.fullmatch(r'\s*(?:ESI\s*[-:]?\s*)?([1-5])(?:\.0+)?\s*', str(esi), re.IGNORECASE)
    if bc is None or esi is None or isinstance(esi, bool) or match is None:
        return None
    return bc, int(match.group(1))


def assign(value: float, task: str):
    """Frozen benchmark-v1 category and out-of-support flag."""
    if task == 'CTD':
        return int(value), False
    centers = CENTERS[task]
    rd = min(range(5), key=lambda i: abs(value - centers[i])) + 1
    lo, hi = SUPPORT[task]
    return rd, not (lo <= value <= hi)


def fit_kmeans_centers(values, task: str, k: int = 5, seed: int = 42):
    """Pooled 1-D K-means centers in R_D order (DNC descending SI, FIP ascending share)."""
    import numpy as np
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(np.asarray(values, float).reshape(-1, 1))
    centers = sorted(km.cluster_centers_.ravel().tolist())
    return centers[::-1] if task == 'DNC' else centers


# ── files ───────────────────────────────────────────────────────────────────

def infer_task(path: Path, payload: dict) -> str:
    for key in ('experiment', 'session'):
        value = str(payload.get(key, '')).upper()
        for task in ('DNC', 'FIP', 'CTD'):
            if value.startswith(task):
                return task
    session = str(payload.get('session', ''))
    if session.startswith('Invest'):
        return 'FIP'
    if session.startswith('BatteryLayers'):
        return 'DNC'
    name = path.name.upper()
    for task in ('DNC', 'FIP', 'CTD'):
        if name.startswith(task):
            return task
    raise ValueError(f'Cannot infer task for {path}')


def score_file(path: Path):
    payload = json.loads(path.read_text(encoding='utf-8'))
    task = infer_task(path, payload)
    model = payload.get('model') or path.stem
    rows = []
    for index, trial in enumerate(payload.get('trials', [])):
        got = extract(trial, task)
        if got is None:
            continue
        bc, raw = got
        rd, flag = assign(raw, task)
        rows.append(dict(task=task, model=model, source=path.name, trial=index, contextual_belief=bc,
                         raw_decision=raw, risk_decision=rd, out_of_support=flag,
                         stored_risk_decision=trial.get('risk_decision', '')))
    return task, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='+', type=Path, help='Phase-2 JSON files (dataset or browser downloads)')
    ap.add_argument('--out', type=Path, default=OUTPUT_DIR / 'experiments' / 'phase2_scored.csv')
    ap.add_argument('--refit', action='store_true', help='also fit pooled K-means centers on these files')
    args = ap.parse_args()

    rows, pooled = [], defaultdict(list)
    for path in args.files:
        task, scored = score_file(path)
        rows += scored
        pooled[task] += [r['raw_decision'] for r in scored]
    if not rows:
        sys.exit('No scorable trials found.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    cells = defaultdict(Counter)
    for r in rows:
        cells[(r['task'], r['model'])][r['risk_decision']] += 1
    print(f'{"task":5s} {"model":24s} {"n":>4s}  R_D counts 1..5   out-of-support')
    for (task, model), counts in sorted(cells.items()):
        oos = sum(r['out_of_support'] for r in rows if r['task'] == task and r['model'] == model)
        print(f'{task:5s} {model:24s} {sum(counts.values()):4d}  '
              f'{" ".join(f"{counts.get(k, 0):3d}" for k in range(1, 6))}   {oos}')
    print(f'{sum(len(c) == 5 for c in cells.values())}/{len(cells)} cells use all five categories')
    if args.refit:
        for task, values in pooled.items():
            if task in CENTERS:
                fitted = fit_kmeans_centers(values, task)
                print(f'{task} pooled K-means centers (R_D order): {[round(c, 6) for c in fitted]}')
                print(f'{task} frozen benchmark-v1 centers     : {[round(c, 6) for c in CENTERS[task]]}')
    print(f'Scored trials written to {args.out}')


if __name__ == '__main__':
    main()
