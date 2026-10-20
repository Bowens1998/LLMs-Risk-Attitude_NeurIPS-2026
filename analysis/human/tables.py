"""LaTeX tables for the human reference (rows in paper order: DNC, CTD, FIP)."""

ORDER = ('DNC', 'CTD', 'FIP')


def primary_table(points, cluster_summary):
    """Table "Primary-cohort task-pooled human AUC": point estimate and participant-cluster
    percentile interval. Formatted to sit below a figure (minipage with a table caption)."""
    rows = []
    for task in ORDER:
        p = points.set_index('task').loc[task]
        s = cluster_summary.set_index('task').loc[task]
        rows.append(f'{task} & {int(p.n_participants)} & {int(p.n_trials)} & {p.auc:.2f} [{s.auc_lo:.2f}, {s.auc_hi:.2f}] \\\\')
    n_boot = int(cluster_summary['successful_draws'].min())
    return ('\\par\\medskip\n\\begin{minipage}{\\linewidth}\n\\makeatletter\\def\\@captype{table}\\makeatother\n'
            '\\centering\\small\n'
            f'\\caption{{Primary-cohort task-pooled human AUC; percentile 95\\% intervals from {n_boot:,} '
            'participant-cluster bootstrap draws.}\\label{tab:human_primary_summary}\n'
            '\\begin{tabular}{lrrr}\n\\toprule\nTask & Participants & Trials & AUC [95\\% CI] \\\\\n\\midrule\n'
            + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n\\end{minipage}\n')


def random_intercept_table(clmm):
    """Table "Human belief--decision slopes from participant-random-intercept cumulative-logit models"."""
    rows = []
    for task in ORDER:
        d = clmm.set_index('task').loc[task]
        rows.append(f'{task} & {int(d.n_participants)} & {int(d.n_trials)} & {d.beta_z:.3f} [{d.ci_low:.3f}, {d.ci_high:.3f}] \\\\')
    return ('\\begin{table}[H]\n\\centering\\small\n'
            '\\caption{Human belief--decision slopes from participant-random-intercept cumulative-logit models. '
            '$B_C$ is standardized within each task; brackets give Wald 95\\% intervals.}\\label{tab:cr_human_clmm}\n'
            '\\begin{tabular}{lrrr}\\toprule Task & Participants & Trials & Standardized slope [95\\% CI] \\\\\\midrule\n'
            + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}\n\\end{table}\n')
