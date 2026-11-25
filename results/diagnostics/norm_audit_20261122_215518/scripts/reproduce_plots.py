from pathlib import Path
import sys
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / 'src/btl_k03').is_dir())
OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from btl_k03.models import build_model_from_config
from btl_k03.utils import set_global_seed

checks = pd.read_csv(OUT / 'tables' / 'checkpoint_checks_and_correlations.csv')
mapping = pd.read_csv(ROOT / 'data/processed/diginetica/item_mapping.csv').sort_values('internal_item_id')
for index, row in checks.iterrows():
    checkpoint = torch.load(ROOT / f'runs/{row.model}/seed_{row.seed}/checkpoint.pth', map_location='cpu', weights_only=False)
    set_global_seed(int(row.seed))
    model = build_model_from_config(checkpoint['config'], mapping, torch.device('cpu'))
    ids = torch.tensor(mapping.loc[mapping.is_controlled_cold, 'internal_item_id'].to_numpy().copy())
    difference = (model.item_embedding.weight.detach()[ids] - checkpoint['model_state_dict']['item_embedding.weight'][ids]).abs().max()
    checks.loc[index, 'cold_initialization_reconstruction_max_abs_error'] = float(difference)
    checks.loc[index, 'cold_initialization_reconstructed_within_1e-7'] = bool(difference <= 1e-7)
checks.to_csv(OUT / 'tables' / 'checkpoint_checks_and_correlations.csv', index=False)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                     'axes.spines.top': False, 'axes.spines.right': False})
models = ['gru4rec', 'narm', 'sasrec']
labels = ['GRU4Rec', 'NARM', 'SASRec']
colors = ['#008545', '#3985AD', '#D68A20']
fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
x = np.arange(3)
for key, label, shift, color in [
    ('rho_frequency_norm_warm', 'Warm, norm gốc', -.25, colors[0]),
    ('rho_frequency_norm_all_including_cold', 'Gộp cả cold', 0, colors[1]),
    ('rho_frequency_norm_centered_warm', 'Warm, trừ tâm warm', .25, colors[2]),
]:
    values = checks.groupby('model')[key].mean().reindex(models).to_numpy()
    bars = ax.bar(x + shift, values, .23, color=color, label=label, zorder=3)
    ax.bar_label(bars, labels=[f'{v:+.3f}' for v in values], padding=5)
ax.axhline(0, color='#666666', lw=1)
ax.set(xticks=x, xticklabels=labels, ylabel='Spearman: train frequency ↔ norm', ylim=(-.95, .4),
       title='Tương quan thay đổi theo phạm vi item và gốc tọa độ')
ax.legend(frameon=False, loc='lower left', ncol=3, fontsize=10)
ax.grid(axis='y', alpha=.18, zorder=0)
fig.savefig(OUT / 'figures' / 'scope_and_origin.png', dpi=200)
fig.savefig(OUT / 'figures' / 'scope_and_origin.svg')
plt.close(fig)

items = pd.read_parquet(OUT / 'data' / 'item_geometry.parquet')
data = items.loc[items.model.eq('sasrec') & items.seed.eq(2024) & items.group.ne('cold')]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout='constrained')
for ax, key, title, ylabel in [
    (axes[0], 'norm', 'Norm âm theo frequency: ρ = −0,7114', 'Norm embedding'),
    (axes[1], 'mean_session_projection', 'Chiếu có dấu dương theo frequency: ρ = +0,7920', 'Chiếu lên hướng phiên trung bình'),
]:
    for group, color in zip(['popular', 'medium', 'rare'], colors):
        sub = data.loc[data.group.eq(group)]
        ax.scatter(sub.frequency, sub[key], s=5, alpha=.22, color=color,
                   edgecolors='none', label=group.capitalize(), rasterized=True)
    ax.set(xscale='log', xlabel='Tần suất train (thang log)', ylabel=ylabel, title=title)
    ax.grid(alpha=.14)
    legend = ax.legend(frameon=False, fontsize=10, markerscale=4)
    for handle in legend.legend_handles:
        handle.set_alpha(1)
fig.savefig(OUT / 'figures' / 'norm_vs_signed_projection.png', dpi=200)
fig.savefig(OUT / 'figures' / 'norm_vs_signed_projection.svg')
plt.close(fig)

verdict_path = OUT / 'audit_verdict.json'
verdict = json.loads(verdict_path.read_text())
verdict['all_cold_initializations_reproduced_within_1e-7'] = bool(checks['cold_initialization_reconstructed_within_1e-7'].all())
verdict['initialization_rounding_note'] = 'SASRec differs only at float32 roundoff (<1.2e-8); GRU4Rec/NARM match exactly.'
verdict_path.write_text(json.dumps(verdict, indent=2), encoding='utf-8')
print('Plots saved; initialization reproduction checked to float32 tolerance.')
