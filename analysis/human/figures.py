"""Human-reference figures.

plot_pooled_olr  - task-pooled human OLR curves over the six frontier-model curves
                   (Appendix figure "Primary task-pooled human OLR curves").
plot_grouped     - grouped descriptive display: human behavioral groups (LOESS with bootstrap
                   band, binned means +- SEM) and the LLM ensemble reference
                   (main-text figure "Human behavioral groups and the LLM reference").
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.lines as mlines  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from statsmodels.nonparametric.smoothers_lowess import lowess  # noqa: E402

from extract import MODELS  # noqa: E402
from fit import GRID, fit_olr  # noqa: E402
from groups import GROUP_NAMES  # noqa: E402

TASK_COLORS = {'DNC': '#d62728', 'CTD': '#1f77b4', 'FIP': '#2ca02c'}
LIGHT_COLORS = {'DNC': '#f5b0b0', 'CTD': '#aecde8', 'FIP': '#b7e4b7'}


def plot_pooled_olr(human, llm, stem, png=True):
    """Solid human segments span the observed beliefs, dashed segments extrapolate; rugs show
    every retained human belief."""
    style = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 8,
             'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
             'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.05), sharey=True, layout='constrained')
        for ax, task in zip(axes, ('DNC', 'CTD', 'FIP')):
            for model in MODELS:
                g = llm[(llm['task'] == task) & (llm['model'] == model)]
                ax.plot(GRID * 100, fit_olr(g['bc'], g['rd']).expected, color='.72', lw=.8, alpha=.6)
            g = human[human['task'] == task]
            curve = fit_olr(g['bc'], g['rd']).expected
            inside = (GRID * 100 >= g['bc'].min()) & (GRID * 100 <= g['bc'].max())
            ax.plot(GRID * 100, curve, '--', color=TASK_COLORS[task], alpha=.4, lw=1)
            ax.plot(GRID[inside] * 100, curve[inside], color=TASK_COLORS[task], lw=1.8)
            ax.plot(g['bc'], np.full(len(g), .83), '|', color=TASK_COLORS[task], alpha=.18, ms=3)
            ax.set(title=task, xlabel='$B_C$', xlim=(0, 100), ylim=(.7, 5.1), xticks=[0, 50, 100], yticks=[1, 3, 5])
        axes[0].set_ylabel('Expected decision')
        fig.savefig(f'{stem}.pdf', bbox_inches='tight')
        if png:
            fig.savefig(f'{stem}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)


def _loess_band(x, y, seed, frac=.55, n_boot=300):
    """LOESS fit and percentile 95% band from trial-bootstrap refits."""
    order = np.argsort(x)
    fit = lowess(y[order], x[order], frac=frac, it=3, return_sorted=True)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        xb, yb = x[idx], y[idx]
        ob = np.argsort(xb)
        sm = lowess(yb[ob], xb[ob], frac=frac, it=2, return_sorted=True)
        boot.append(np.interp(fit[:, 0], sm[:, 0], sm[:, 1]))
    boot = np.array(boot)
    return fit[:, 0], fit[:, 1], np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)


def ensemble_curve(params, classes, x):
    """E[Y | x] for logit P(Y <= k) = theta_k - beta x."""
    beta = params.loc['beta']
    theta = params.drop('beta').to_numpy(float)
    cum = 1 / (1 + np.exp(-(theta[None, :] - beta * x[:, None])))
    probs = np.diff(np.hstack([np.zeros((len(x), 1)), cum, np.ones((len(x), 1))]), axis=1)
    return probs @ np.asarray(classes, float)


def plot_grouped(grouped, path, ensemble=None, llm=None):
    """Rows CTD/FIP/DNC; columns are the groups ordered by their mean outcome in that task.
    ensemble: R output (task, parameter, value) for the LLM reference, or None to omit it."""
    rows = ('CTD', 'FIP', 'DNC')
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.subplots_adjust(hspace=0.08, wspace=0.08)
    x_ref = np.linspace(0, 100, 200)
    for r, task in enumerate(rows):
        data = grouped[grouped['task'] == task]
        color, light = TASK_COLORS[task], LIGHT_COLORS[task]
        columns = data.groupby('group')['y'].mean().sort_values(kind='stable').index
        reference = None
        if ensemble is not None:
            params = ensemble[ensemble['task'] == task].set_index('parameter')['value']
            classes = np.unique(llm.loc[llm['task'] == task, 'rd'])
            reference = ensemble_curve(params, classes, x_ref / 100)
        for c, group in enumerate(columns):
            ax = axes[r, c]
            sub = data[data['group'] == group]
            x, y = sub['bc'].to_numpy(), sub['y'].to_numpy()
            if r == 0:
                ax.set_title(f'{GROUP_NAMES[c]} Group', fontsize=26, fontweight='bold', color=color, pad=8)
            ax.set_xticks([0, 100])
            ax.set_yticks([1, 5])
            if r == 2:
                ax.set_xlabel('$B_C$', fontsize=26)
                ax.tick_params(axis='x', labelsize=40)
            else:
                ax.tick_params(axis='x', labelbottom=False)
            if c == 0:
                ax.set_ylabel('$R_D$', fontsize=26)
                ax.tick_params(axis='y', labelsize=40)
            else:
                ax.tick_params(axis='y', labelleft=False)
            ax.set(xlim=(-5, 105), ylim=(.3, 5.7))
            ax.grid(True, alpha=.25, linestyle='--')
            for lo in range(0, 100, 20):
                v = y[(x >= lo) & (x < lo + 20)]
                if len(v) >= 3:
                    ax.errorbar(lo + 10, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), fmt='D', color=color,
                                markersize=7, linewidth=1.5, capsize=3, zorder=4)
            if len(x) >= 10:
                xs, ys, lo_band, hi_band = _loess_band(x, y, seed=r * 3 + c)
                ax.fill_between(xs, lo_band, hi_band, color=light, alpha=.55, zorder=3)
                ax.plot(xs, ys, color='black', linewidth=3.5, zorder=5)
                ax.plot(xs, ys, color=color, linewidth=2.0, zorder=6)
            if reference is not None:
                ax.plot(x_ref, reference, color='black', linewidth=2.0, linestyle='--', zorder=7)
    fig.canvas.draw()
    for c in range(3):  # keep "100" from running into the next panel's "0"
        axes[2, c].xaxis.get_major_ticks()[-1].label1.set_ha('right')
    handles = [mpatches.Patch(facecolor=LIGHT_COLORS[t], edgecolor=TASK_COLORS[t], linewidth=1.5, label=f'Human {t}')
               for t in rows]
    if reference is not None:
        handles.append(mlines.Line2D([], [], color='black', linewidth=2.5, linestyle='--', label='LLM Ensemble Baseline'))
    legend = fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=20, framealpha=.9,
                        handlelength=3, handleheight=1.5, borderpad=.8, columnspacing=2.0, bbox_to_anchor=(.5, -.06))
    legend.get_frame().set_linewidth(0)
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
