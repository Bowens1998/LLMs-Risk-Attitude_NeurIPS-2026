"""Offline check of the runner against the published Base/Instruct Phase-1 runs (no API calls).

Each per-run sidecar (*_meta.json) in data/hf/FrontierModel_reliability_N90/ stores the frozen
scenario and every recorded model response of each valid trial. This script feeds those
responses back through ic_runner.py (the API call is replaced by the recorded reply) and
compares the resulting B_C (all tasks), R_D (CTD, DNC) and, for DNC, the full drone
trajectory with the published records. FIP R_D in these runs was scored with fixed cut-offs
on the high-risk weight rather than the runner's nearest-center rule, so for FIP only B_C is
compared. It also checks the sidecar scenarios against
experiments/config/phase1_frozen_scenarios.json.

Usage:
    python experiments/runner/replay_check.py
"""
import contextlib
import copy
import hashlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from riskattitude.paths import FRONTIER_RELIABILITY, REPO_ROOT, require_data  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ic_runner as R  # noqa: E402

PRESETS = {'FIP': R.FIP_PRESETS, 'CTD': R.CTD_PRESETS, 'DNC': R.DNC_PRESETS}
BUNDLE = REPO_ROOT / 'experiments' / 'config' / 'phase1_frozen_scenarios.json'


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode('utf-8')).hexdigest()


def replay_trial(task, frozen, level, raw_trial):
    """Return (B_C, R_D, trajectory_ok) obtained from the recorded responses."""
    if task == 'FIP':
        ctx, _, _, _ = R._validate_fip_response(raw_trial)
        return ctx, None, True
    replies = [s['raw'] for s in raw_trial['steps']]
    if task == 'DNC':
        replies.append(raw_trial['survey_raw'])
    replies = iter(replies)
    live_call = R.call_llm
    R.call_llm = lambda *args, **kwargs: copy.deepcopy(next(replies))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            scenario = R._thaw_scenario(task, frozen)
            if task == 'CTD':
                result = R.run_ctd_trial(scenario, PRESETS[task][level], {'api_mode': 'chat'})
            else:
                result = R.run_dnc_trial(scenario, level, PRESETS[task][level], {'api_mode': 'chat'})
    finally:
        R.call_llm = live_call
    if result is None:
        return None, None, False
    path_ok = True
    if task == 'DNC':
        path_ok = ([(s['row'], s['col']) for s in raw_trial['steps']]
                   == [(s['row'], s['col']) for s in result['steps']])
    return result['ctx'], result['sb'], path_ok


def main():
    require_data()
    bundle = json.loads(BUNDLE.read_text(encoding='utf-8'))['experiments']
    sidecars = sorted(FRONTIER_RELIABILITY.glob('*/*/*/*/*_meta.json'))
    if not sidecars:
        sys.exit('No per-run *_meta.json sidecars found; run data/download_data.py without --skip-meta.')
    print(f'{"run":48s} {"protocol":44s} {"B_C":>7s} {"R_D":>7s} {"path":>7s}  scenarios')
    for meta_path in sidecars:
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        task = meta['experiment']
        records = json.loads(meta_path.with_name(meta_path.name.replace('_meta.json', '.json'))
                             .read_text(encoding='utf-8'))
        n = bc_ok = rd_ok = path_ok = 0
        same_scenarios = True
        for cfg in meta['configs']:
            level = int(cfg['preset_level'])
            same_scenarios &= _hash(cfg['frozen_scenario']) == bundle[task]['levels'][str(level)]['sha256']
            level_records = [r for r in records if r['level'] == level]
            for rec, raw_trial in zip(level_records, cfg['raw_trials'], strict=True):
                ctx, rd, ok = replay_trial(task, cfg['frozen_scenario'], level, raw_trial)
                n += 1
                bc_ok += ctx is not None and float(ctx) == float(rec['contextual_belief'])
                rd_ok += rd is not None and int(rd) == int(rec['risk_decision'])
                path_ok += ok
        protocol = meta.get('protocol_version') or meta.get('protocol') or '-'
        rd_col = '      -' if task == 'FIP' else f'{rd_ok:>3d}/{n:<3d}'
        print(f'{meta_path.stem.replace("_meta", ""):48s} {protocol:44s} {bc_ok:>3d}/{n:<3d} '
              f'{rd_col} {path_ok:>3d}/{n:<3d}  {"bundle" if same_scenarios else "DIFFERENT"}')


if __name__ == '__main__':
    main()
