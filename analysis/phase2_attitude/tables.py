"""LaTeX tables in the layout used by the paper."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .trials import DISPLAY_ORDER, EARLIER_LABELS, FRONTIER_LABELS


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding='utf-8')


def _p_text(p: float) -> str:
    return r'$<.001$' if p < 0.001 else f'{p:.3f}'.lstrip('0')


def _with_ci(point: float, summary: pd.DataFrame, task: str, model: str, metric: str) -> str:
    b = summary[(summary.task == task) & (summary.model == model) & (summary.metric == metric)].iloc[0]
    return f'{point:.3f} [{b.q025:.3f}, {b.q975:.3f}]'


def uncertainty_tables(points: pd.DataFrame, summary: pd.DataFrame, n_boot: int, out: Path) -> None:
    """Main-text table (with Wald p) and the appendix variant without p."""
    rows_main, rows_app, last = [], [], None
    for p in points.sort_values(['task', 'model']).itertuples():
        auc = _with_ci(p.auc, summary, p.task, p.model, 'auc')
        beta = _with_ci(p.beta, summary, p.task, p.model, 'beta')
        if last is not None and p.task != last:
            rows_main.append(r'\midrule')
        last = p.task
        label = FRONTIER_LABELS[p.model]
        rows_main.append(f'{p.task} & {label} & {auc} & {beta} & {_p_text(p.beta_p)} \\\\')
        rows_app.append(f'{p.task} & {label} & {auc} & {beta} \\\\')
    nb = f'{n_boot:,}'
    _write(out / 'main_uncertainty_table.tex', '\n'.join([
        r'\begin{table}[!tbp]', r'\centering\small',
        r'\caption{\textbf{Risk attitude and sensitivity estimates.} AUC and $\hat\beta$ with percentile '
        rf'95\% intervals from {nb} trial-bootstrap refits per model--task cell ($N=100$), alongside '
        r'Wald $p_\beta$ values.}\label{tab:cr_uncertainty}\label{tab:main_slope_intervals}',
        r'\setlength{\tabcolsep}{3pt}', r'\renewcommand{\arraystretch}{0.98}',
        r'\begin{tabular}{@{}llrrr@{}}', r'\toprule',
        r'Task & Model & AUC [95\% CI] & $\hat\beta$ [95\% CI] & Wald $p_\beta$ \\', r'\midrule',
        *rows_main, r'\bottomrule', r'\end{tabular}', r'\end{table}', '']))
    _write(out / 'uncertainty_table.tex', '\n'.join([
        r'\begin{table}[htbp]', r'\centering\small',
        rf'\caption{{Point estimates and percentile intervals from {nb} trial-bootstrap refits per '
        r'model--task cell ($N=100$). Extreme CTD slope intervals signal quasi-separation; they are '
        r'retained without truncation.}\label{tab:cr_uncertainty}',
        r'\begin{tabular}{llrr}', r'\toprule',
        r'Task & Model & AUC [95\% CI] & $\beta$ [95\% CI] \\', r'\midrule',
        *rows_app, r'\bottomrule', r'\end{tabular}', r'\end{table}', '']))


def diagnostics_table(diag: pd.DataFrame, out: Path) -> None:
    rows = [f'{d.task} & {FRONTIER_LABELS[d.model]} & {d.bc_min:.0f}--{d.bc_max:.0f} & '
            f'{d.rd1}/{d.rd2}/{d.rd3}/{d.rd4}/{d.rd5} & {d.mean_rd:.2f} & {d.auc_support:.2f} \\\\'
            for d in diag.itertuples()]
    _write(out / 'diagnostics_table.tex', '\n'.join([
        r'\begin{table}[H]', r'\centering\small',
        r'\caption{Observed belief support, response-category counts and support-restricted mean curve '
        r"height. $\mathrm{AUC}_{S}=|S|^{-1}\int_S E(x)dx$ uses each cell's own support, so its "
        r'integration domain differs across cells.}\label{tab:cr_diagnostics}',
        r'\begin{tabular}{llcrrr}', r'\toprule',
        r'Task & Model & $B_C$ range & Counts $R_D=1/2/3/4/5$ & Mean $R_D$ & $\mathrm{AUC}_{S}$ \\',
        r'\midrule', *rows, r'\bottomrule', r'\end{tabular}', r'\end{table}', '']))


def point_rank_tables(diag: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Six-model AUC point ranks (tab:risk_rank) and AUC / mean-decision ranks (tab:cr_mean_ranks)."""
    ranks = {}
    for task, g in diag.groupby('task'):
        auc_rank = g.auc.rank(method='average')
        mean_rank = g.mean_rd.rank(method='average')
        for model, a, m in zip(g.model, auc_rank, mean_rank):
            ranks[task, model] = (int(a), int(m))
    tasks = ('DNC', 'CTD', 'FIP')
    lines = [f'{FRONTIER_LABELS[m]} & ' + ' & '.join(str(ranks[t, m][0]) for t in tasks) + r' \\'
             for m in DISPLAY_ORDER]
    _write(out / 'risk_rank_table.tex', '\n'.join([
        r'\begin{table}[ht]\centering\small',
        r'\caption{Descriptive point-estimate ranks of six models (1 = lowest AUC). These are not '
        r'uncertainty-resolved orderings.}\label{tab:risk_rank}',
        r'\begin{tabular}{lccc}\toprule Model & DNC & CTD & FIP \\',
        r'\midrule ' + lines[0], *lines[1:], r'\bottomrule\end{tabular}\end{table}', '']))
    n_diff = sum(a != m for a, m in ranks.values())
    lines = [f'{FRONTIER_LABELS[m]} & ' + ' & '.join(f'{ranks[t, m][0]} / {ranks[t, m][1]}' for t in tasks)
             + r' \\' for m in DISPLAY_ORDER]
    _write(out / 'mean_decision_rank_table.tex', '\n'.join([
        r'\begin{table}[H]', r'\centering\small',
        r'\caption{AUC rank / raw mean-decision rank for all six models (1 = lowest value for both '
        rf'measures). Ranks differ in {n_diff}/{len(ranks)} model--task cells.}}\label{{tab:cr_mean_ranks}}',
        r'\begin{tabular}{lccc}\toprule Model & DNC & CTD & FIP \\\midrule',
        *lines, r'\bottomrule\end{tabular}', r'\end{table}', '']))
    return pd.DataFrame([{'task': t, 'model': m, 'auc_rank': a, 'mean_rd_rank': r}
                         for (t, m), (a, r) in ranks.items()])


def pairwise_ordering_table(task_orders: dict[str, list[str]], out: Path) -> None:
    rows = [f'{task} & ' + r' $>$ '.join(FRONTIER_LABELS[m] for m in order) + r' \\'
            for task, order in task_orders.items()]
    _write(out / 'pairwise_ordering_table.tex', '\n'.join([
        r'\begin{table}[H]', r'\centering\small',
        r'\caption{Pairwise-probability AUC orderings from the bootstrap distributions.}'
        r'\label{tab:pairwise_ordering}',
        r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{@{}ll@{}}', r'\toprule',
        r'Task & Pairwise-probability ordering (higher $\rightarrow$ lower AUC) \\', r'\midrule',
        *rows, r'\bottomrule', r'\end{tabular}', r'\end{table}', '']))


def earlier_rank_table(points: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Earlier-model AUC ranks ordered by mean rank (tab:rank_earlier)."""
    tasks = ('DNC', 'FIP', 'CTD')
    models = list(dict.fromkeys(points.model))
    ranks = {}
    for task in tasks:
        g = points[points.task == task]
        ordered = sorted(models, key=lambda m: float(g.loc[g.model == m, 'auc'].iloc[0]))
        for i, m in enumerate(ordered, start=1):
            ranks[task, m] = i
    table = pd.DataFrame([{'model': m, **{t: ranks[t, m] for t in tasks}} for m in models])
    table['mean_rank'] = table[list(tasks)].mean(axis=1)
    table['rank_sd'] = table[list(tasks)].std(axis=1, ddof=0)
    table = table.sort_values('mean_rank', kind='stable').reset_index(drop=True)
    flagged = table.loc[table.rank_sd.idxmax(), 'model']
    sd = table.rank_sd.max()
    lines = []
    for r in table.itertuples():
        name = EARLIER_LABELS[r.model] + (r'$^\dagger$' if r.model == flagged else '')
        lines.append(f'{name} & {r.DNC} & {r.FIP} & {r.CTD} & {r.mean_rank:.1f} \\\\')
    _write(out / 'earlier_rank_table.tex', '\n'.join([
        r'\begin{table}[htbp]', r'\centering',
        r'\caption{Risk attitude ranks of earlier-generation models across three tasks (rank 1 = most '
        r'cautious, rank 6 = most aggressive; ordered by mean rank). $^\dagger$Highest rank standard '
        rf'deviation ($\sigma_r = {sd:.2f}$).}}', r'\label{tab:rank_earlier}', r'\small',
        r'\setlength{\tabcolsep}{8pt}', r'\begin{tabular}{lcccc}', r'\toprule',
        r'\textbf{Model} & \textbf{DNC} & \textbf{FIP} & \textbf{CTD} & \textbf{Mean rank} \\',
        r'\midrule', *lines, r'\bottomrule', r'\end{tabular}', r'\end{table}', '']))
    return table
