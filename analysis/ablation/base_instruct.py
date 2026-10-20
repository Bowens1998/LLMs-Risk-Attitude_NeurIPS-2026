"""Matched Base -> Instruct comparisons (paper Appendix, "Matched Base--Instruct Comparisons").

Inputs (public dataset, data/hf/FrontierModel_reliability_N90/):
  * Reliability_<TASK>_merged.json  -- valid Phase-1 records (analysis input)
  * per-run *_meta.json sidecars    -- attempted trials per scenario, frozen-scenario
                                       hashes and decoding settings (completion yield)

Invalid generations are completion failures; they are never imputed as behaviour,
so all behavioural statistics are conditional on valid completion.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from riskattitude.paths import FRONTIER_RELIABILITY

TASK_ORDER = ('DNC', 'CTD', 'FIP')          # row order of the completion/reliability table
DISPERSION_ORDER = ('FIP', 'CTD', 'DNC')    # comparison order of the dispersion table and Fisher tests
FAMILIES = {
    'Gemma': {'base': 'Gemma4E4b', 'instruct': 'Gemma4E4bIt'},
    'Qwen': {'base': 'Qwen330bA3bBase', 'instruct': 'Qwen330bA3bInstruct2507'},
}
SERVED_MODEL_IDS = {
    'google/gemma-4-E4B': 'Gemma4E4b',
    'google/gemma-4-E4B-it': 'Gemma4E4bIt',
    'Qwen/Qwen3-30B-A3B-Base': 'Qwen330bA3bBase',
    'Qwen/Qwen3-30B-A3B-Instruct-2507': 'Qwen330bA3bInstruct2507',
}
P1_MAX, P2_MIN = 20.0, 80.0
RNG_SEED = 42
# Seed blocks used for the published intervals/p-values: comparison i (0..5, in
# DISPERSION_ORDER per family) uses RNG_SEED + offset + 1 + i.
SEED_OFFSETS = {('sd', 'bootstrap'): 0, ('sd', 'permutation'): 100,
                ('entropy', 'bootstrap'): 200, ('entropy', 'permutation'): 300}


@dataclass
class Cell:
    """One checkpoint x task cell."""
    model_key: str
    task: str
    records: pd.DataFrame
    attempts_by_level: dict = field(default_factory=dict)   # empty if sidecars are unavailable
    scenario_hashes: dict = field(default_factory=dict)
    settings: tuple | None = None

    @property
    def valid(self) -> int:
        return int(len(self.records))

    @property
    def attempts(self) -> int | None:
        return int(sum(self.attempts_by_level.values())) if self.attempts_by_level else None


# ── loading ──────────────────────────────────────────────────────────────────

def _canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _find_sidecars(task: str) -> dict[str, Path]:
    """Map model_key -> per-run sidecar for one task (searched below FRONTIER_RELIABILITY)."""
    found = {}
    for path in sorted(FRONTIER_RELIABILITY.glob(f'*/{task}/*/*/{task}_*_meta.json')):
        with open(path, encoding='utf-8') as handle:
            model_id = json.load(handle).get('model')
        key = SERVED_MODEL_IDS.get(model_id)
        if key is None:
            continue
        if key in found:
            raise ValueError(f'Two sidecars for {key}/{task}: {found[key]} and {path}')
        found[key] = path
    return found


def load_cells() -> tuple[dict[tuple[str, str], Cell], bool]:
    """Return {(model_key, task): Cell} and whether all sidecars were found."""
    cells, complete = {}, True
    keys = [k for fam in FAMILIES.values() for k in fam.values()]
    for task in TASK_ORDER:
        merged = pd.DataFrame(json.loads(
            (FRONTIER_RELIABILITY / f'Reliability_{task}_merged.json').read_text(encoding='utf-8')))
        sidecars = _find_sidecars(task)
        for key in keys:
            records = merged[(merged['model_key'] == key) & (merged['experiment'] == task)].reset_index(drop=True)
            if records.empty:
                raise ValueError(f'No records for {key}/{task} in the merged reliability file')
            cell = Cell(key, task, records)
            path = sidecars.get(key)
            if path is None:
                complete = False
            else:
                meta = json.loads(path.read_text(encoding='utf-8'))
                configs = sorted(meta['configs'], key=lambda c: int(c['preset_level']))
                cell.attempts_by_level = {int(c['preset_level']): int(c['attempts']) for c in configs}
                cell.scenario_hashes = {int(c['preset_level']): _canonical_hash(c['frozen_scenario'])
                                        for c in configs}
                inference = meta.get('inference_config') or {}
                cell.settings = (meta.get('seed'), inference.get('api_mode'), inference.get('temperature'),
                                 inference.get('top_p'), inference.get('max_tokens'),
                                 meta.get('max_extra_attempts'))
                # The per-run file must hold exactly the merged records (same order).
                run_records = json.loads(path.with_name(path.name.replace('_meta.json', '.json'))
                                         .read_text(encoding='utf-8'))
                if run_records != records.to_dict('records'):
                    raise ValueError(f'Per-run records differ from merged records for {key}/{task}')
                valid_by_level = records.groupby('level').size().to_dict()
                for level, n_attempts in cell.attempts_by_level.items():
                    if valid_by_level.get(level, 0) > n_attempts:
                        raise ValueError(f'{key}/{task} L{level}: more valid records than attempts')
            cells[(key, task)] = cell
    return cells, complete


# ── per-scenario and task-level metrics ─────────────────────────────────────

def normalized_entropy(values) -> float:
    counts = np.bincount(np.asarray(values, dtype=int), minlength=6)[1:6]
    p = counts[counts > 0] / counts.sum()
    return max(0.0, -float(np.sum(p * np.log(p))) / math.log(5))


def scenario_metrics(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level, group in records.groupby('level', sort=True):
        bc = group['contextual_belief'].to_numpy(float)
        rd = group['risk_decision'].to_numpy(int)
        mean = float(bc.mean())
        sd = float(bc.std(ddof=1)) if len(bc) > 1 else 0.0
        rsd = 0.0 if mean == 0 and sd == 0 else (100 * sd / mean if mean > 0 else math.nan)
        counts = pd.Series(rd).value_counts()
        rows.append(dict(level=int(level), n=len(bc), bc_mean=mean, bc_sd=sd, rsd_percent=rsd,
                         rd_mode=int(counts[counts == counts.max()].index.min()),
                         modal_share_percent=100 * counts.max() / len(rd),
                         rd_entropy=normalized_entropy(rd)))
    return pd.DataFrame(rows)


def task_summary(cell: Cell) -> dict:
    s = scenario_metrics(cell.records)
    p1, p2 = float(s['rsd_percent'].mean()), float(s['modal_share_percent'].mean())
    return dict(p1=p1, p2=p2, task_pass=bool(p1 <= P1_MAX and p2 >= P2_MIN))


def within_condition_sd(records: pd.DataFrame) -> float:
    """sqrt(sum of squared deviations from each scenario mean / total within-scenario df)."""
    residuals, df = [], 0
    for _, group in records.groupby('level', sort=True):
        values = group['contextual_belief'].to_numpy(float)
        residuals.extend(values - values.mean())
        df += max(0, len(values) - 1)
    return math.sqrt(float(np.sum(np.asarray(residuals) ** 2)) / df) if df > 0 else math.nan


def mean_condition_entropy(records: pd.DataFrame) -> float:
    return float(np.mean([normalized_entropy(g['risk_decision'].to_numpy())
                          for _, g in records.groupby('level', sort=True)]))


METRICS = {'sd': within_condition_sd, 'entropy': mean_condition_entropy}


# ── resampling ──────────────────────────────────────────────────────────────

def _entropy_rows(draws: np.ndarray, n: int) -> np.ndarray:
    p = np.stack([(draws == k).sum(axis=1) for k in range(1, 6)], axis=1) / n
    safe = np.where(p > 0, p, 1)
    return -np.sum(np.where(p > 0, p * np.log(safe), 0), axis=1) / math.log(5)


def _bootstrap_draws(records: pd.DataFrame, metric: str, n_boot: int, rng) -> np.ndarray:
    """Condition-stratified bootstrap distribution of one checkpoint's metric."""
    groups = [g for _, g in records.groupby('level', sort=True)]
    if metric == 'sd':
        ss, df = np.zeros(n_boot), 0
        for g in groups:
            values = g['contextual_belief'].to_numpy(float)
            s = values[rng.integers(0, len(values), size=(n_boot, len(values)))]
            ss += np.sum((s - s.mean(axis=1, keepdims=True)) ** 2, axis=1)
            df += len(values) - 1
        return np.sqrt(ss / df)
    entropies = []
    for g in groups:
        values = g['risk_decision'].to_numpy(int)
        entropies.append(_entropy_rows(values[rng.integers(0, len(values), size=(n_boot, len(values)))],
                                       len(values)))
    return np.mean(np.stack(entropies, axis=1), axis=1)


def bootstrap_ci(base: pd.DataFrame, instruct: pd.DataFrame, metric: str, n_boot: int, seed: int):
    """Percentile 95% interval for Instruct - Base (Instruct resampled first)."""
    rng = np.random.default_rng(seed)
    diff = _bootstrap_draws(instruct, metric, n_boot, rng) - _bootstrap_draws(base, metric, n_boot, rng)
    lo, hi = np.quantile(diff, [0.025, 0.975])
    return float(lo), float(hi)


def permutation_p(base: pd.DataFrame, instruct: pd.DataFrame, metric: str, n_perm: int, seed: int) -> float:
    """Two-sided p for Instruct - Base; labels are permuted within each scenario."""
    rng = np.random.default_rng(seed)
    observed = METRICS[metric](instruct) - METRICS[metric](base)
    column = 'contextual_belief' if metric == 'sd' else 'risk_decision'
    acc = {'base': [], 'instruct': []}
    ss = {'base': np.zeros(n_perm), 'instruct': np.zeros(n_perm)}
    df = {'base': 0, 'instruct': 0}
    for level in sorted(set(base['level']) | set(instruct['level'])):
        b = base.loc[base['level'] == level, column].to_numpy()
        a = instruct.loc[instruct['level'] == level, column].to_numpy()
        combined = np.concatenate([b, a])
        order = np.argsort(rng.random((n_perm, len(combined))), axis=1)
        draws = {'base': combined[order[:, :len(b)]], 'instruct': combined[order[:, len(b):]]}
        for name, d in draws.items():
            if metric == 'sd':
                d = d.astype(float)
                ss[name] += np.sum((d - d.mean(axis=1, keepdims=True)) ** 2, axis=1)
                df[name] += d.shape[1] - 1
            else:
                acc[name].append(_entropy_rows(d, d.shape[1]))
    if metric == 'sd':
        permuted = np.sqrt(ss['instruct'] / df['instruct']) - np.sqrt(ss['base'] / df['base'])
    else:
        permuted = (np.mean(np.stack(acc['instruct'], axis=1), axis=1)
                    - np.mean(np.stack(acc['base'], axis=1), axis=1))
    return float((1 + np.sum(np.abs(permuted) >= abs(observed))) / (n_perm + 1))


def holm(pvalues) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    adjusted, running = np.empty(len(p)), 0.0
    for rank, index in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[index]))
        adjusted[index] = running
    return adjusted.tolist()


# ── tables ──────────────────────────────────────────────────────────────────

def comparison_pairs():
    """(index, family, task) in the order that defines the resampling seeds and Holm families."""
    pairs = [(fam, task) for fam in FAMILIES for task in DISPERSION_ORDER]
    return [(i, fam, task) for i, (fam, task) in enumerate(pairs)]


def reliability_table(cells) -> pd.DataFrame:
    rows = []
    for fam, keys in FAMILIES.items():
        for task in TASK_ORDER:
            b, a = cells[(keys['base'], task)], cells[(keys['instruct'], task)]
            sb, sa = task_summary(b), task_summary(a)
            rows.append(dict(family=fam, task=task,
                             base_valid=b.valid, base_attempted=b.attempts,
                             instruct_valid=a.valid, instruct_attempted=a.attempts,
                             base_p1=sb['p1'], instruct_p1=sa['p1'],
                             base_p2=sb['p2'], instruct_p2=sa['p2'],
                             base_pass=sb['task_pass'], instruct_pass=sa['task_pass']))
    return pd.DataFrame(rows)


def completion_table(cells) -> pd.DataFrame:
    rows = []
    for _, fam, task in comparison_pairs():
        b, a = cells[(FAMILIES[fam]['base'], task)], cells[(FAMILIES[fam]['instruct'], task)]
        if b.attempts is None or a.attempts is None:
            continue
        odds, p = fisher_exact([[a.valid, a.attempts - a.valid], [b.valid, b.attempts - b.valid]],
                               alternative='two-sided')
        rows.append(dict(family=fam, task=task,
                         base_valid=b.valid, base_attempted=b.attempts, base_rate=b.valid / b.attempts,
                         instruct_valid=a.valid, instruct_attempted=a.attempts,
                         instruct_rate=a.valid / a.attempts,
                         odds_ratio=odds, fisher_p=p,
                         same_scenarios=b.scenario_hashes == a.scenario_hashes,
                         same_settings=b.settings == a.settings))
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame['fisher_p_holm'] = holm(frame['fisher_p'])
    return frame


def dispersion_table(cells, n_boot: int, n_perm: int, seed: int = RNG_SEED) -> pd.DataFrame:
    rows = []
    for i, fam, task in comparison_pairs():
        b = cells[(FAMILIES[fam]['base'], task)].records
        a = cells[(FAMILIES[fam]['instruct'], task)].records
        row = dict(family=fam, task=task)
        for metric, fn in METRICS.items():
            lo, hi = bootstrap_ci(b, a, metric, n_boot, seed + SEED_OFFSETS[(metric, 'bootstrap')] + 1 + i)
            p = permutation_p(b, a, metric, n_perm, seed + SEED_OFFSETS[(metric, 'permutation')] + 1 + i)
            row.update({f'{metric}_base': fn(b), f'{metric}_instruct': fn(a),
                        f'{metric}_delta': fn(a) - fn(b), f'{metric}_ci_low': lo, f'{metric}_ci_high': hi,
                        f'{metric}_p': p})
        rows.append(row)
    frame = pd.DataFrame(rows)
    for metric in METRICS:
        frame[f'{metric}_p_holm'] = holm(frame[f'{metric}_p'])
    return frame


def condition_table(cells) -> pd.DataFrame:
    frames = []
    for (key, task), cell in cells.items():
        s = scenario_metrics(cell.records)
        s.insert(0, 'task', task)
        s.insert(0, 'model_key', key)
        s.insert(3, 'attempted', s['level'].map(cell.attempts_by_level) if cell.attempts_by_level else np.nan)
        frames.append(s)
    return pd.concat(frames, ignore_index=True)


# ── LaTeX in the paper's layout ─────────────────────────────────────────────

def _yes(flag: bool) -> str:
    return 'Yes' if flag else 'No'


def reliability_latex(table: pd.DataFrame) -> str:
    def count(v, n):
        return f'{v}/{n}' if pd.notna(n) else f'{v}/--'
    rows = [f'{r.family}/{r.task} & {count(r.base_valid, r.base_attempted)} $\\to$ '
            f'{count(r.instruct_valid, r.instruct_attempted)} & {r.base_p1:.1f} $\\to$ {r.instruct_p1:.1f} & '
            f'{r.base_p2:.1f} $\\to$ {r.instruct_p2:.1f} & {_yes(r.base_pass)} $\\to$ {_yes(r.instruct_pass)} \\\\'
            for r in table.itertuples()]
    return ('\\begin{table}[H]\n\\centering\\small\n'
            '\\caption{Matched Base $\\to$ Instruct results, conditional on valid completion for P1/P2. '
            'Task passage requires mean P1$\\leq20\\%$ and mean P2$\\geq80\\%$, with equal weight for each of '
            'the three scenarios, as in the main analysis.}\\label{tab:cr_ablation}\n'
            '\\begin{tabular}{lcccc}\n\\toprule\n'
            'Family/task & Valid/attempted & P1 (\\%) & P2 (\\%) & Task pass \\\\\n\\midrule\n'
            + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')


def _p_text(p: float) -> str:
    return f'{p:.4f}' if p < 0.001 else f'{p:.3f}'


def dispersion_latex(table: pd.DataFrame) -> str:
    def cell(r, metric, places):
        g = lambda name: getattr(r, f'{metric}_{name}')  # noqa: E731
        return ('\\begin{tabular}[t]{@{}l@{}}'
                f'{g("base"):.{places}f} $\\to$ {g("instruct"):.{places}f}\\\\'
                f'$\\Delta={g("delta"):+.{places}f}$ [{g("ci_low"):.{places}f}, {g("ci_high"):.{places}f}]\\\\'
                f'$p_H={_p_text(g("p_holm"))}$\\end{{tabular}}')
    rows = [f'{r.family}/{r.task} & {cell(r, "sd", 2)} & {cell(r, "entropy", 3)} \\\\'
            for r in table.itertuples()]
    return ('\\begin{table}[H]\n\\centering\\small\n'
            '\\caption{\\textbf{Matched Base--Instruct changes in behavioral dispersion.} Values show Base '
            '$\\to$ Instruct; $\\Delta$ is Instruct minus Base, with 95\\% bootstrap intervals and Holm-adjusted '
            'permutation $p_H$. Completion counts and the common P1/P2 task gate are reported in '
            'Table~\\ref{tab:cr_ablation}.}\\label{tab:cr_ablation_dispersion}\n'
            '\\setlength{\\tabcolsep}{5pt}\n\\begin{tabular}{@{}lll@{}}\n\\toprule\n'
            'Family/task & Within-condition $B_C$ SD & Normalized $R_D$ entropy \\\\\n\\midrule\n'
            + '\n\\addlinespace[4pt]\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')


def summary_lines(rel: pd.DataFrame, comp: pd.DataFrame, disp: pd.DataFrame, n_boot: int, n_perm: int) -> list[str]:
    """Plain-text statements that accompany the two tables in the paper."""
    out = [f'Resamples: {n_boot:,} bootstrap, {n_perm:,} permutation (seed {RNG_SEED}).', '']
    out.append(f'Task passes: Base {int(rel.base_pass.sum())}/6, Instruct {int(rel.instruct_pass.sum())}/6 ('
               + ', '.join(f'{r.family}/{r.task}' for r in rel.itertuples() if r.instruct_pass) + ').')
    for side in ('base', 'instruct'):
        for fam in FAMILIES:
            sub = rel[rel.family == fam]
            out.append(f'  {fam} {side}: {int(sub[f"{side}_pass"].sum())}/3 tasks pass '
                       f'(overall gate: {"pass" if sub[f"{side}_pass"].all() else "fail"}).')
    out.append(f'Modal share increases: {int((rel.instruct_p2 > rel.base_p2).sum())}/6; '
               f'mean RSD decreases: {int((rel.instruct_p1 < rel.base_p1).sum())}/6.')
    for metric, label in (('sd', 'within-condition B_C SD'), ('entropy', 'normalized R_D entropy')):
        sig = disp[(disp[f'{metric}_p_holm'] < 0.05)]
        dec = sig[sig[f'{metric}_delta'] < 0]
        out.append(f'Significant (p_H<0.05) reductions in {label}: {len(dec)}/6 '
                   f'(p_H values {sorted({round(x, 4) for x in dec[f"{metric}_p_holm"]})}); '
                   f'significant increases: {len(sig) - len(dec)}.')
    if not comp.empty:
        ctd_dnc = comp[comp.task.isin(['CTD', 'DNC'])]
        fip = comp[comp.task == 'FIP']
        out.append(f'Completion (Fisher, Holm over 6): CTD/DNC max p_H = {ctd_dnc.fisher_p_holm.max():.3g}; '
                   f'FIP p_H = {", ".join(f"{p:.3f}" for p in fip.fisher_p_holm)}.')
        out.append(f'Max attempts in any checkpoint-task cell: {int(max(comp.base_attempted.max(), comp.instruct_attempted.max()))}.')
    return out
