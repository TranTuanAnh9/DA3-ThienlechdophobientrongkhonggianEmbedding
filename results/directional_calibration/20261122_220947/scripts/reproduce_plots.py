from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / 'src/btl_k03').is_dir())
OUT = Path(__file__).resolve().parents[1]
selection = pd.read_csv(OUT / 'tables' / 'selection_grid.csv')
confirmation = pd.read_csv(OUT / 'tables' / 'confirmation_comparison.csv')
intervals = pd.read_csv(OUT / 'tables' / 'confirmation_bootstrap.csv')
summary = json.loads((OUT / 'summary.json').read_text())
policies = json.loads((OUT / 'locked_policies.json').read_text())
names = {'direction': 'Hướng phiên', 'constant_direction': 'Hướng phiên cố định',
         'matched_random': 'Hướng ngẫu nhiên', 'popularity': 'Penalty popularity'}
colors = {'direction': '#008545', 'constant_direction': '#3985AD',
          'matched_random': '#999999', 'popularity': '#D68A20'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})
base = selection.loc[selection.method.eq('baseline')].groupby('seed').mean(numeric_only=True)
selection_deltas = []
for (method, coefficient), rows in selection.loc[selection.method.ne('baseline')].groupby(['method', 'coefficient']):
    delta = rows.set_index('seed').select_dtypes('number') - base.select_dtypes('number')
    selection_deltas.append({'method': method, 'coefficient': coefficient,
        'overall_recall_delta_pp': delta['recall@20'].mean() * 100,
        'rare_recall_delta_pp': delta['rare_recall@20'].mean() * 100,
        'rare_mrr_delta_pp': delta['rare_mrr@20'].mean() * 100})
grid = pd.DataFrame(selection_deltas)
grid.to_csv(OUT / 'tables' / 'selection_deltas.csv', index=False)
fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
for method in names:
    data = grid.loc[grid.method.eq(method)].sort_values('coefficient')
    ax.plot(data.overall_recall_delta_pp, data.rare_recall_delta_pp, 'o-',
            label=names[method], color=colors[method], alpha=.9)
    winner = next(p['coefficient'] for p in policies['policies'] if p['method'] == method)
    if winner:
        point = data.loc[data.coefficient.eq(winner)].iloc[0]
        ax.scatter([point.overall_recall_delta_pp], [point.rare_recall_delta_pp], marker='*', s=180,
                   color=colors[method], edgecolor='black', linewidth=.8, zorder=4)
ax.axvline(0, color='#555555', lw=1); ax.axhline(0, color='#555555', lw=1)
ax.set(title='Chọn hệ số trên nửa validation theo session', xlabel='Δ Recall tổng thể (điểm %)',
       ylabel='Δ Rare Recall (điểm %)')
ax.grid(alpha=.16); ax.legend(frameon=False)
fig.savefig(OUT / 'figures' / 'selection_tradeoff.png', dpi=200)
fig.savefig(OUT / 'figures' / 'selection_tradeoff.svg')
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout='constrained')
ordered = [(method, seed) for method in names if method != 'matched_random' for seed in (2024, 2025)]
for ax, group, metric, title in [(axes[0], 'rare', 'hit', 'Rare Recall'), (axes[1], 'all_warm', 'hit', 'Recall tổng thể')]:
    for index, (method, seed) in enumerate(ordered):
        row = intervals.loc[intervals.treatment.eq(method) & intervals.control.eq('baseline')
                            & intervals.seed.eq(seed) & intervals.group.eq(group) & intervals.metric.eq(metric)].iloc[0]
        ax.errorbar(row.delta * 100, index,
                    xerr=[[100 * (row.delta - row.ci_low)], [100 * (row.ci_high - row.delta)]],
                    fmt='o', capsize=4, color=colors[method])
    ax.axvline(0, ls='--', color='#666666')
    ax.set(yticks=np.arange(len(ordered)), yticklabels=[f'{names[m]} — {s}' for m, s in ordered],
           title=title, xlabel='Chênh lệch so với baseline (điểm %)')
    ax.invert_yaxis(); ax.grid(axis='x', alpha=.18)
fig.suptitle('Kiểm tra hệ số đã khóa trên nửa validation còn lại — CI bootstrap 95%', fontsize=14)
fig.savefig(OUT / 'figures' / 'confirmation_intervals.png', dpi=200)
fig.savefig(OUT / 'figures' / 'confirmation_intervals.svg')
plt.close(fig)

rows = []
for seed in (2024, 2025):
    baseline = confirmation.loc[confirmation.seed.eq(seed) & confirmation.method.eq('baseline')].iloc[0]
    for row in confirmation.loc[confirmation.seed.eq(seed)].to_dict('records'):
        row.update({f'delta_{key}_pp': (row[key] - baseline[key]) * 100
                    for key in ('recall@20', 'mrr@20', 'rare_recall@20', 'rare_mrr@20', 'coverage@20')})
        row['delta_gini'] = row['gini@20'] - baseline['gini@20']
        rows.append(row)
pd.DataFrame(rows).to_csv(OUT / 'tables' / 'confirmation_deltas.csv', index=False)
mean_delta = pd.DataFrame(rows).groupby('method').mean(numeric_only=True)
print('HELD-OUT VALIDATION DELTAS\n', mean_delta.filter(regex='delta_').to_string())
for seed in (2024, 2025):
    baseline = pd.read_csv(OUT / 'predictions' / f'seed_{seed}/baseline_confirmation_per_query.csv.gz')
    for method in names:
        query = pd.read_csv(OUT / 'predictions' / f'seed_{seed}/{method}_confirmation_per_query.csv.gz')
        assert len(query) == summary['validation_confirmation_queries']
        row = confirmation.loc[confirmation.seed.eq(seed) & confirmation.method.eq(method)].iloc[0]
        assert np.isclose(query.hit.mean(), row['recall@20'])
        assert np.isclose(query.loc[query.group.eq('rare'), 'hit'].mean(), row['rare_recall@20'])
        assert np.isclose(query.loc[query.group.eq('rare'), 'reciprocal_rank'].mean(), row['rare_mrr@20'])
summary['per_query_metrics_verified'] = True
summary['interpretation_scope'] = 'exploratory internal validation confirmation; no independent test claim'
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print('SUMMARY', json.dumps(summary))
