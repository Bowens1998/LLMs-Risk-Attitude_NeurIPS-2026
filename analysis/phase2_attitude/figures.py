"""Phase-2 figures: fitted OLR curves and AUC / slope bar charts."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .trials import DISPLAY_ORDER, EARLIER_LABELS, FRONTIER_LABELS

matplotlib.use('Agg')

CURVE_COLOR = {'DNC': '#c0271f', 'CTD': '#1a6fab', 'FIP': '#1a8c1a'}
BAND_COLOR = {'DNC': '#f4a9a8', 'CTD': '#aacfee', 'FIP': '#a6dba6'}
BAR_COLOR = {'DNC': '#E07070', 'CTD': '#5B9BD5', 'FIP': '#70B870'}
BAR_EDGE = {'DNC': '#c04040', 'CTD': '#3a7abf', 'FIP': '#3a8a3a'}
SUPPORT_COLOR = {'DNC': '#d62728', 'CTD': '#1f77b4', 'FIP': '#2ca02c'}
TASK_ORDER = ('DNC', 'CTD', 'FIP')
BAND_SEED = 42


def _value(points: pd.DataFrame, task: str, model: str, metric: str) -> float:
    return float(points.loc[(points.task == task) & (points.model == model), metric].iloc[0])


def olr_curves_by_model(fits: dict, out: Path, n_draws: int = 300) -> None:
    """2x3 panels, one per model: fitted E[R_D | B_C] per task with approximate
    pointwise 95% bands from multivariate-normal coefficient draws."""
    rng = np.random.default_rng(BAND_SEED)
    grid = np.linspace(0, 100, 201)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True, sharex=True)
    fig.subplots_adjust(hspace=0.38, wspace=0.08)
    for ax, model in zip(axes.flat, DISPLAY_ORDER):
        for task in TASK_ORDER:
            fit = fits[task, model]
            curve = fit.expected(grid / 100)
            lo, hi = fit.coefficient_band(grid / 100, rng, n_draws)
            ax.plot(grid, curve, color=CURVE_COLOR[task], linewidth=2.0, zorder=4)
            ax.fill_between(grid, lo, hi, color=BAND_COLOR[task], alpha=0.40, zorder=3)
        ax.set_title(FRONTIER_LABELS[model], fontsize=21, fontweight='bold')
        ax.set(xlim=(-2, 102), ylim=(0.7, 5.3), yticks=[1, 5], xticks=[0, 100])
        ax.tick_params(axis='both', labelsize=18)
        ax.spines[['top', 'right']].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$R_D$', fontsize=19)
    for ax in axes[1, :]:
        ax.set_xlabel(r'$B_C$', fontsize=19)
    handles = [mpatches.Patch(color=CURVE_COLOR[t], label=t) for t in TASK_ORDER]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=18, framealpha=0.85,
               bbox_to_anchor=(0.5, -0.10))
    fig.savefig(out / 'olr_scurves_by_model.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def _grouped_bars(ax, order, values, human=None, offsets=(-0.26, 0, 0.26), width=0.26):
    for i, model in enumerate(order):
        for off, task in zip(offsets, TASK_ORDER):
            ax.bar(i + off, values[task, model], width, color=BAR_COLOR[task], alpha=0.85, zorder=3)
        mean = np.mean([values[t, model] for t in TASK_ORDER])
        ax.hlines(mean, i - 1.5 * width, i + 1.5 * width, colors='#1a1a1a', linewidths=1.6,
                  linestyles='dashed', zorder=6)
    if human is not None:
        i = len(order)
        for off, task in zip(offsets, TASK_ORDER):
            ax.bar(i + off, human[task], width, color=BAR_COLOR[task], alpha=0.5, hatch='//',
                   edgecolor=BAR_EDGE[task], zorder=3)
        mean = np.mean([human[t] for t in TASK_ORDER])
        ax.hlines(mean, i - 1.5 * width, i + 1.5 * width, colors='#1a1a1a', linewidths=1.6,
                  linestyles='dashed', zorder=6)
        ax.axhline(mean, color='gray', linestyle='--', linewidth=1.0, alpha=0.7, zorder=2)
    n = len(order) + (human is not None)
    ax.set_xlim(-0.55, n - 0.45)
    ax.set_xticks(np.arange(n) - width)
    ax.spines[['top', 'right']].set_visible(False)


def attitude_sensitivity_bars(points: pd.DataFrame, out: Path, failed=('Grok4',),
                              human: pd.DataFrame | None = None) -> None:
    """AUC (top) and slope (bottom) per model and task; dashed segments are task means.

    `human` (optional) has columns task, auc, beta and is drawn as a hatched reference group.
    """
    auc = {(t, m): _value(points, t, m, 'auc') for t in TASK_ORDER for m in DISPLAY_ORDER}
    beta = {(t, m): _value(points, t, m, 'beta') for t in TASK_ORDER for m in DISPLAY_ORDER}
    h_auc = h_beta = None
    if human is not None:
        h_auc = dict(zip(human.task, human.auc))
        h_beta = dict(zip(human.task, human.beta))
    beta_order = sorted(DISPLAY_ORDER, key=lambda m: np.mean([beta[t, m] for t in TASK_ORDER]),
                        reverse=True)

    def names(order):
        labels = [FRONTIER_LABELS[m] + (' *' if m in failed else '') for m in order]
        return labels + (['Human'] if human is not None else [])

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(15, 16), gridspec_kw={'hspace': 0.75})
    _grouped_bars(top, DISPLAY_ORDER, auc, h_auc)
    top.set_xticklabels(names(DISPLAY_ORDER), fontsize=27, rotation=-45, ha='left', fontweight='bold')
    top.set_ylabel('Risk Attitude', fontsize=26)
    top.set_ylim(1, 4.1)
    top.set_yticks([1, 4])
    top.tick_params(axis='y', labelsize=30)
    top.annotate('← More Risk-Averse', xy=(0.18, 1.02), xycoords='axes fraction', fontsize=26,
                 fontweight='bold', ha='center')
    top.annotate('More Risk-Taking →', xy=(0.70, 1.02), xycoords='axes fraction', fontsize=26,
                 fontweight='bold', ha='center')

    _grouped_bars(bottom, beta_order, beta, h_beta)
    bottom.axhline(0, color='black', linewidth=0.8, zorder=2)
    bottom.set_xticklabels(names(beta_order), fontsize=27, rotation=-45, ha='left', fontweight='bold')
    bottom.set_ylabel(r'Risk Sensitivity ($\beta$)', fontsize=26)
    all_betas = list(beta.values()) + (list(h_beta.values()) if h_beta else [])
    bottom.set_yticks([np.floor(min(all_betas) / 5) * 5, 0])
    bottom.tick_params(axis='y', labelsize=30)
    bottom.annotate('← Less Sensitive', xy=(0.18, 1.02), xycoords='axes fraction', fontsize=26,
                    fontweight='bold', ha='center')
    bottom.annotate('More Sensitive →', xy=(0.70, 1.02), xycoords='axes fraction', fontsize=26,
                    fontweight='bold', ha='center')

    handles = [mpatches.Patch(color=BAR_COLOR[t], alpha=0.85, label=t) for t in TASK_ORDER]
    handles.append(plt.Line2D([0], [0], color='#1a1a1a', linewidth=1.6, linestyle='dashed', label='Mean'))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=24,
               framealpha=0.85, handlelength=2.0, handleheight=1.5)
    if failed:
        fig.text(0.5, -0.11, '* Fail to pass the reliability test.', fontsize=20, ha='center')
    fig.align_ylabels([top, bottom])
    fig.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.15)
    fig.savefig(out / 'Histgram_RiskAttitude&RiskSensitivity.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def _earlier_axes(order, labels, ylabel):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    n = len(order)
    for x in np.arange(0.5, n - 0.5):
        ax.axvline(x, color='#cccccc', linewidth=0.6, zorder=1)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlim(-0.55, n - 0.45)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    return fig, ax


def earlier_bars(points: pd.DataFrame, out: Path, failed=None) -> None:
    """Earlier-generation models: AUC ordered by mean AUC; slope ordered by median slope,
    with the axis clipped and clipped bars annotated with their values."""
    tasks = ('DNC', 'FIP', 'CTD')
    models = list(dict.fromkeys(points.model))
    failed = set(models if failed is None else failed)
    val = {(t, m, k): _value(points, t, m, k) for t in tasks for m in models for k in ('auc', 'beta')}
    offsets, w = (-0.26, 0, 0.26), 0.26

    def names(order):
        return [EARLIER_LABELS[m] + (' *' if m in failed else '') for m in order]

    order = sorted(models, key=lambda m: np.mean([val[t, m, 'auc'] for t in tasks]))
    fig, ax = _earlier_axes(order, names(order), 'AUC (Area Under OLR S-Curve)')
    for i, m in enumerate(order):
        for off, t in zip(offsets, tasks):
            ax.bar(i + off, val[t, m, 'auc'], w, color=BAR_COLOR[t], alpha=0.75, zorder=3,
                   edgecolor='white', linewidth=0.5)
        ax.hlines(np.mean([val[t, m, 'auc'] for t in tasks]), i - 1.5 * w, i + 1.5 * w,
                  colors='#1a1a1a', linewidths=1.4, linestyles='dashed', zorder=6)
    ax.annotate('← More Risk-Averse', xy=(0.20, 1.01), xycoords='axes fraction', fontsize=10.5,
                fontweight='bold', ha='center')
    ax.annotate('More Risk-Taking →', xy=(0.76, 1.01), xycoords='axes fraction', fontsize=10.5,
                fontweight='bold', ha='center')
    handles = [mpatches.Patch(color=BAR_COLOR[t], alpha=0.85, label=f'$B_C$→$R_D$ in {t}') for t in tasks]
    handles.append(plt.Line2D([0], [0], color='#1a1a1a', linewidth=1.4, linestyle='dashed', label='Mean AUC'))
    ax.legend(handles=handles, loc='upper left', fontsize=9.5, framealpha=0.85)
    if failed:
        ax.text(0.22, 0.95, '* Fail to pass the test.', transform=ax.transAxes, fontsize=8.5, style='italic')
    fig.tight_layout()
    fig.savefig(out / 'AUC_Barchart_AllModels.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

    betas = np.array([val[t, m, 'beta'] for t in tasks for m in models])
    q1, q3 = np.percentile(betas, [25, 75])
    clip = (max(q1 - 2 * (q3 - q1), -40), 2)
    order = sorted(models, key=lambda m: np.median([val[t, m, 'beta'] for t in tasks]))
    fig, ax = _earlier_axes(order, names(order), 'β Coefficient (OLR Risk Sensitivity)')
    for i, m in enumerate(order):
        raw = [val[t, m, 'beta'] for t in tasks]
        for off, t, v in zip(offsets, tasks, raw):
            shown = min(max(v, clip[0]), clip[1])
            ax.bar(i + off, shown, w, color=BAR_COLOR[t], alpha=0.85, zorder=3, edgecolor='white',
                   linewidth=0.5)
            if shown != v:
                ax.annotate(f'{v:.0f}', xy=(i + off, shown), xytext=(0, -10), textcoords='offset points',
                            ha='center', va='top', fontsize=7.5, color='#c04040', fontweight='bold')
        ax.hlines(min(max(np.mean(raw), clip[0]), clip[1]), i - 1.5 * w, i + 1.5 * w,
                  colors='#1a1a1a', linewidths=1.4, linestyles='dashed', zorder=6)
    ax.set_ylim(clip[0] * 1.15, clip[1] + abs(clip[1]) * 0.2 + 0.5)
    ax.axhline(0, color='gray', linewidth=0.8, alpha=0.5)
    ax.text(len(order) - 0.5, clip[0] * 1.08, 'clipped values shown in red', fontsize=8, ha='right',
            color='#c04040', style='italic')
    handles = [mpatches.Patch(color=BAR_COLOR[t], alpha=0.85, label=f'β in {t}') for t in tasks]
    handles.append(plt.Line2D([0], [0], color='#1a1a1a', linewidth=1.4, linestyle='dashed', label='Mean β'))
    ax.legend(handles=handles, loc='upper left', fontsize=9.5, framealpha=0.85)
    if failed:
        ax.text(0.17, 0.95, '* Fail to pass the test.', transform=ax.transAxes, fontsize=8.5, style='italic')
    fig.tight_layout()
    fig.savefig(out / 'Beta_Barchart_AllModels.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def observed_support(fits: dict, frontier: pd.DataFrame, out: Path) -> None:
    """Fitted curves with solid segments inside each cell's observed B_C range,
    dashed extrapolation outside it, and belief rugs."""
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7, 'pdf.fonttype': 42})
    fig, axs = plt.subplots(2, 3, figsize=(5.5, 3.65), sharex=True, sharey=True, layout='constrained')
    x = np.linspace(0, 1, 1001)
    for ax, model in zip(axs.flat, DISPLAY_ORDER):
        for j, task in enumerate(TASK_ORDER):
            fit = fits[task, model]
            bc = frontier[(frontier.task == task) & (frontier.model == model)].bc.to_numpy()
            y = fit.expected(x)
            inside = (x >= bc.min() / 100) & (x <= bc.max() / 100)
            ax.plot(x * 100, y, ls='--', color=SUPPORT_COLOR[task], alpha=0.4, lw=0.9)
            ax.plot(x[inside] * 100, y[inside], color=SUPPORT_COLOR[task], lw=1.3, label=task)
            ax.plot(bc, np.full(len(bc), 0.72 + j * 0.08), '|', color=SUPPORT_COLOR[task], alpha=0.35, ms=3)
        ax.set(title=FRONTIER_LABELS[model], xlim=(0, 100), ylim=(0.64, 5.1), xticks=[0, 50, 100],
               yticks=[1, 3, 5])
        ax.spines[['top', 'right']].set_visible(False)
    for ax in axs[-1]:
        ax.set_xlabel('Contextual belief $B_C$')
    for ax in axs[:, 0]:
        ax.set_ylabel('Expected decision')
    axs[0, 0].legend(loc='upper right', fontsize=6, frameon=False)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'supplement_observed_support.{ext}', bbox_inches='tight', dpi=300)
    plt.close(fig)
    plt.rcdefaults()
