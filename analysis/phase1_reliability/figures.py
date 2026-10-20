"""Five-panel consistency figure for one model-task cell (appendix reliability figures).

A: B_C by scenario (boxes: IQR, lines: medians); B: B_C against R_D; C: mean B_C with
approximate 95% mean intervals (mean +/- 1.96 s / sqrt(n)); D: R_D category proportions;
E: scenario metrics and the task-average gate. Jitter (seed 42) is for display only.
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from metrics import P1_MAX, P2_MIN  # noqa: E402

STYLE = {
    'font.family': ['Arial', 'DejaVu Sans'], 'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
}
SCENARIO_COLORS = {1: '#3b82f6', 2: '#22c55e', 3: '#ef4444'}
RD_COLORS = {1: '#1e3a8a', 2: '#3b82f6', 3: '#94a3b8', 4: '#f59e0b', 5: '#dc2626'}


def _verdict(ok):
    return 'PASS' if ok else 'FAIL'


def plot_consistency(result, out_paths):
    with plt.rc_context(STYLE):
        _plot_consistency(result, out_paths)


def _plot_consistency(result, out_paths):
    scenarios, p1, p2 = result['scenarios'], result['p1'], result['p2']
    fig = plt.figure(figsize=(9.6, 7.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, .7], hspace=.58, wspace=.32, left=.085, right=.98, top=.95, bottom=.07)
    aa, ab = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ac, ad = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    ae = fig.add_subplot(gs[2, :])
    rng = np.random.default_rng(42)

    for s in scenarios:
        c, bc, rd, n = s['scenario'], s['bc'], s['rd'], s['n']
        i = c - 1
        aa.scatter(c + rng.uniform(-.18, .18, n), bc, color=SCENARIO_COLORS[c], s=23, alpha=.72,
                   edgecolors='white', linewidths=.3, label=f'C{c}', zorder=3)
        q1, q3 = np.percentile(bc, [25, 75])
        med = float(np.median(bc))
        aa.add_patch(Rectangle((c - .22, q1), .44, q3 - q1, fill=False, edgecolor='#1e293b', linewidth=1.1, zorder=4))
        aa.plot([c - .22, c + .22], [med, med], color='#1e293b', lw=1.5, zorder=4)
        ab.scatter(bc, rd + rng.uniform(-.13, .13, n), color=SCENARIO_COLORS[c], s=23, alpha=.72,
                   edgecolors='white', linewidths=.3, label=f'C{c}', zorder=3)
        ci = 1.96 * s['sd'] / np.sqrt(n)
        ac.barh(i, s['mean'], xerr=ci, color=SCENARIO_COLORS[c], alpha=.82, height=.5,
                error_kw={'ecolor': '#333', 'capsize': 3, 'linewidth': 1.1}, zorder=3)
        ac.text(117, i + .28, f'RSD {s["rsd"]:.1f}%', ha='right', va='center', fontsize=10, color='#222')
        left = 0.
        for k in range(1, 6):
            prop = float(np.mean(rd == k))
            if prop:
                ad.barh(i, prop, left=left, color=RD_COLORS[k], height=.5, alpha=.85, zorder=3)
                if prop >= .13:
                    ad.text(left + prop / 2, i, str(k), ha='center', va='center', color='white', weight='bold', fontsize=10)
            left += prop
        ad.text(.99, i + .30, f'Modal {s["modal"]:.1f}%', ha='right', va='center', fontsize=10)

    aa.set(xlim=(.4, 3.6), ylim=(-5, 105), xticks=[1, 2, 3], xticklabels=['C1', 'C2', 'C3'],
           xlabel='Condition', ylabel='Contextual belief ($B_C$)')
    aa.axhline(50, color='#aaa', lw=.7, ls='--')
    aa.grid(axis='y', alpha=.25)
    aa.legend(fontsize=9, loc='upper left', framealpha=.8)
    ab.set(xlim=(-2, 102), ylim=(.5, 5.5), yticks=[1, 2, 3, 4, 5], xlabel='Contextual belief ($B_C$)', ylabel='Risk decision ($R_D$)')
    ab.grid(alpha=.25)
    ac.set(yticks=[0, 1, 2], yticklabels=[f'C{s["scenario"]} (n={s["n"]})' for s in scenarios],
           xlim=(0, 120), ylim=(-.65, 2.6), xlabel='Contextual belief ($B_C$)')
    ac.axvline(50, color='#aaa', lw=.7, ls='--')
    ac.grid(axis='x', alpha=.25)
    ad.set(xlim=(0, 1), ylim=(-.65, 2.6), yticks=[0, 1, 2], yticklabels=['C1', 'C2', 'C3'], xlabel='Proportion of trials')
    ad.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ad.grid(axis='x', alpha=.25)
    ad.legend(handles=[Patch(color=RD_COLORS[k], label=str(k)) for k in range(1, 6)], title='$R_D$', ncol=5, fontsize=8,
              title_fontsize=9, loc='lower center', bbox_to_anchor=(.5, -.48), frameon=False, handlelength=1, columnspacing=.8)
    for ax, label in zip([aa, ab, ac, ad], 'ABCD'):
        ax.set_title(label, loc='left', weight='bold', pad=6)

    ae.axis('off')
    ae.set_title('E', loc='left', weight='bold', pad=4)
    rows = [[f'C{s["scenario"]}', str(s['n']), f'{s["mean"]:.1f} ± {s["sd"]:.1f}', f'{s["rsd"]:.1f}', str(s['mode']),
             f'{s["modal"]:.1f}', _verdict(s['rsd'] <= P1_MAX), _verdict(s['modal'] >= P2_MIN)] for s in scenarios]
    rows.append(['Task mean', str(result['n']), '—', f'{p1:.1f}', '—', f'{p2:.1f}', _verdict(result['p1_pass']), _verdict(result['p2_pass'])])
    headers = ['Scenario', 'n', '$B_C$ mean ± SD', 'RSD (%)', 'Mode $R_D$', 'Modal (%)', 'P1 ≤20%', 'P2 ≥80%']
    tab = ae.table(cellText=rows, colLabels=headers, cellLoc='center', bbox=[0, .22, 1, .75],
                   colWidths=[.12, .06, .21, .12, .12, .13, .12, .12])
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    for j in range(8):
        tab[0, j].set_facecolor('#1e293b')
        tab[0, j].set_text_props(color='white', weight='bold')
    for i in range(1, len(rows) + 1):
        for j in range(8):
            cell = tab[i, j]
            cell.set_facecolor('#f8fafc' if i % 2 else '#f1f5f9')
            if j >= 6:
                ok = rows[i - 1][j] == 'PASS'
                cell.set_facecolor('#dcfce7' if ok else '#fee2e2')
                cell.set_text_props(color='#15803d' if ok else '#b91c1c', weight='bold')
    ae.text(.5, .07, f'Task gate: mean P1 = {p1:.1f}%, mean P2 = {p2:.1f}%  →  ' + _verdict(result['passed']),
            ha='center', va='center', weight='bold', fontsize=11,
            color='#15803d' if result['passed'] else '#b91c1c', transform=ae.transAxes)

    for path in out_paths:
        fig.savefig(path, bbox_inches='tight', **({'dpi': 300} if str(path).endswith('.png') else {}))
    plt.close(fig)
