"""LaTeX and CSV tables for the Phase-1 reliability results."""
import csv

from metrics import EARLIER_MODELS, FRONTIER_MODELS, P1_MAX, P2_MIN

TASKS_MAIN = ('DNC', 'CTD', 'FIP')          # column order of the main-text tables
TASKS_EARLIER = ('DNC', 'FIP', 'CTD')       # column order of the earlier-model table
CTX_ORDER = ['ClaudeSonnet4.5', 'DeepSeekV3.2', 'GPT5.2', 'Qwen3Max', 'Gemini3Pro', 'Grok4']
RD_ORDER = ['ClaudeSonnet4.5', 'DeepSeekV3.2', 'Gemini3Pro', 'GPT5.2', 'Qwen3Max', 'Grok4']
SCENARIO_LABEL = {'ClaudeSonnet4.5': 'Sonnet 4.5'}


def _task_mean_table(results, metric, order, caption, label, extreme):
    """Rows of task means plus a descriptive across-task mean; the most extreme cell is starred."""
    values = {(m, t): results[(t, m)][metric] for m in order for t in TASKS_MAIN}
    star = extreme(values, key=values.get)
    lines = [r'\begin{table}[t]', r'\centering', rf'\caption{{{caption}}}', rf'\label{{{label}}}', r'\small',
             r'\begin{tabular}{lcccc}', r'\toprule',
             r'\textbf{Model} & \textbf{DNC} & \textbf{CTD} & \textbf{FIP} & \textbf{Mean} \\', r'\midrule']
    for m in order:
        cells = [f'{values[(m, t)]:.1f}' + ('$^*$' if (m, t) == star else '') for t in TASKS_MAIN]
        mean = sum(values[(m, t)] for t in TASKS_MAIN) / len(TASKS_MAIN)
        lines.append(f'{FRONTIER_MODELS[m]} & ' + ' & '.join(cells) + f' & {mean:.1f} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines) + '\n'


def table_ctx_consistency(results):
    caption = (r'\textbf{Contextual belief consistency (mean RSD, \%).} Task entries average three scenario RSDs; '
               r'lower is more repeatable. The final Mean column is descriptive: overall passage requires all three '
               r'tasks to pass, not a threshold on the across-task mean. $^*$Notable outlier.')
    return _task_mean_table(results, 'p1', CTX_ORDER, caption, 'tab:ctx_consistency', max)


def table_rd_consistency(results):
    caption = (r'\textbf{Risk decision consistency (dominant class proportion, \%).} Task entries average three '
               r'scenario modal shares; higher is more repeatable, not necessarily more belief-responsive. '
               r'The final Mean column does not determine passage. $^*$Notable outlier.')
    return _task_mean_table(results, 'p2', RD_ORDER, caption, 'tab:rd_consistency', min)


def table_scenarios(results):
    lines = [r'\begin{table}[htbp]', r'\centering\small',
             r'\caption{Scenario-level reliability metrics; three entries follow scenario order. Equal-weight means of '
             rf'the three unrounded scenario values determine task passage (P1$\leq{P1_MAX:g}\%$ and P2$\geq{P2_MIN:g}\%$).}}'
             r'\label{tab:cr_conditions}',
             r'\begin{tabular}{llrrr}', r'\toprule',
             r'Task & Model & $n_1/n_2/n_3$ & RSD (\%) & Modal share (\%) \\', r'\midrule']
    for m in CTX_ORDER:
        for t in TASKS_MAIN:
            sc = results[(t, m)]['scenarios']
            lines.append(f'{t} & {SCENARIO_LABEL.get(m, m)} & ' + '/'.join(str(s['n']) for s in sc) + ' & '
                         + '/'.join(f'{s["rsd"]:.1f}' for s in sc) + ' & ' + '/'.join(f'{s["modal"]:.1f}' for s in sc) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines) + '\n'


def table_earlier(results):
    mark = lambda ok: r'\cmark' if ok else r'\xmark'
    lines = [r'\begin{table}[htbp]', r'\centering',
             r'\caption{Measurement reliability for earlier-generation models ($N=90$ per model per task). '
             r'\cmark{} = pass, \xmark{} = fail.}', r'\label{tab:ic_earlier}', r'\small',
             r'\begin{tabular}{l ccc ccc ccc}', r'\toprule',
             r'\multirow{2}{*}{\textbf{Model}} & \multicolumn{3}{c}{\textbf{DNC}} & \multicolumn{3}{c}{\textbf{FIP}} '
             r'& \multicolumn{3}{c}{\textbf{CTD}} \\',
             r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}',
             r'& P1 & P2 & Rel & P1 & P2 & Rel & P1 & P2 & Rel \\', r'\midrule']
    for m, name in EARLIER_MODELS.items():
        cells = []
        for t in TASKS_EARLIER:
            r = results[(t, m)]
            cells += [mark(r['p1_pass']), mark(r['p2_pass']), mark(r['passed'])]
        lines.append(f'{name} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines) + '\n'


def write_metrics_csv(path, cohort_results):
    """One row per scenario plus one task-mean row per model-task cell."""
    fields = ['cohort', 'model', 'task', 'scenario', 'n', 'mean_bc', 'sd_bc', 'rsd', 'mode_rd', 'modal_share',
              'p1_pass', 'p2_pass', 'task_pass']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cohort, results in cohort_results.items():
            for (task, model), r in results.items():
                for s in r['scenarios']:
                    w.writerow(dict(cohort=cohort, model=model, task=task, scenario=s['scenario'], n=s['n'],
                                    mean_bc=s['mean'], sd_bc=s['sd'], rsd=s['rsd'], mode_rd=s['mode'], modal_share=s['modal'],
                                    p1_pass=s['rsd'] <= P1_MAX, p2_pass=s['modal'] >= P2_MIN, task_pass=''))
                w.writerow(dict(cohort=cohort, model=model, task=task, scenario='mean', n=r['n'], rsd=r['p1'],
                                modal_share=r['p2'], p1_pass=r['p1_pass'], p2_pass=r['p2_pass'], task_pass=r['passed']))


def write_grid_csv(path, grid):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(grid[0]))
        w.writeheader()
        w.writerows(grid)
