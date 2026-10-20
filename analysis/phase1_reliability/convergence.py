"""Main-text convergence figure: 3 x 3 grid (tasks x scenarios) of B_C against R_D for the
six frontier models, with jittered trials, covariance ellipses and mean markers."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

MODELS = [
    ('ClaudeSonnet4.5', 'Claude Sonnet 4.5', '#1565C0'),
    ('GPT5.2', 'GPT-5.2', '#E65100'),
    ('Qwen3Max', 'Qwen3-Max', '#2E7D32'),
    ('Gemini3Pro', 'Gemini-3 Pro', '#C62828'),
    ('DeepSeekV3.2', 'DeepSeek-V3.2', '#C8A000'),
    ('Grok4', 'Grok-4', '#1A1A1A'),
]
ROWS = [
    ('DNC', 'Drone Navigation\nControl (DNC)', 5),
    ('CTD', 'Clinical Triage\nDecision (CTD)', 3),
    ('FIP', 'Financial Investment\nPortfolio (FIP)', 5),
]
SCENARIO_TITLES = {1: 'Condition 1\n(Low Risk)', 2: 'Condition 2\n(Moderate Risk)', 3: 'Condition 3\n(High Risk)'}
N_STD = 1.645  # ellipse half-axes in SD units


def _ellipse(ax, xs, ys, color):
    if len(xs) < 3:
        return
    eigvals, eigvecs = np.linalg.eigh(np.cov(xs, ys))
    eigvals = np.maximum(eigvals, 1e-6)
    angle = np.degrees(np.arctan2(*eigvecs[:, 1][::-1]))
    minor, major = 2 * N_STD * np.sqrt(eigvals)
    centre = (np.mean(xs), np.mean(ys))
    ax.add_patch(Ellipse(centre, major, minor, angle=angle, facecolor=color, edgecolor='none', alpha=0.15, zorder=2))
    ax.add_patch(Ellipse(centre, major, minor, angle=angle, facecolor='none', edgecolor=color, linewidth=1.8,
                         linestyle='--', alpha=0.9, zorder=3))


def plot_convergence(records, out_path):
    """records: frontier reliability records (Reliability_ALL_merged.json)."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    fig.subplots_adjust(left=0.0696, right=0.910, top=0.950, bottom=0.110, wspace=0.218, hspace=0.278)
    rng = np.random.default_rng(42)
    for row, (task, title, top) in enumerate(ROWS):
        for col, scenario in enumerate((1, 2, 3)):
            ax = axes[row, col]
            sizes = []
            for model, _, color in MODELS:
                pts = [r for r in records if r['experiment'] == task and r['model_key'] == model and r['level'] == scenario]
                if not pts:
                    continue
                sizes.append(len(pts))
                xs = np.array([r['contextual_belief'] for r in pts], float)
                ys = np.array([r['risk_decision'] for r in pts], float)
                ax.scatter(xs + rng.uniform(-1.5, 1.5, len(xs)), ys + rng.uniform(-0.12, 0.12, len(ys)),
                           s=22, color=color, alpha=0.35, linewidths=0, zorder=2)
                _ellipse(ax, xs, ys, color)
                ax.scatter(np.mean(xs), np.mean(ys), marker='D', s=150, color=color, edgecolors='white',
                           linewidths=1.2, zorder=5)
            ax.set_xlim(-5.6, 105.6)
            ax.set_ylim(0.4, top + 0.6)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_yticks(range(1, top + 1))
            ax.tick_params(labelsize=11, color='#888888')
            ax.grid(alpha=0.25, zorder=0)
            for spine in ax.spines.values():
                spine.set_color('#999999')
            n_text = f'N = {min(sizes)} per model' if min(sizes) == max(sizes) else f'N = {min(sizes)}–{max(sizes)} per model'
            ax.text(0.97, 0.04, n_text, transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
                    style='italic', color='gray')
            if row == 0:
                ax.set_title(SCENARIO_TITLES[scenario], fontsize=13, fontweight='bold')
            if row == 2:
                ax.set_xlabel('Contextual Risk Belief ($B_C$)', fontsize=14)
            if col == 0:
                ax.set_ylabel('Risk Decision ($R_D$)', fontsize=14)
        right = axes[row, 2].twinx()
        right.set_yticks([])
        right.spines[:].set_visible(False)
        right.set_ylabel(title, fontsize=13, fontweight='bold', rotation=-90, labelpad=34)
    handles = [Line2D([], [], marker='s', linestyle='', markersize=11, color=c, label=label) for _, label, c in MODELS]
    fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=12, frameon=True, bbox_to_anchor=(0.49, 0.025))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
