"""Fixed-neighbor diagnostic diagrams with saved numerical provenance."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .persistent_homology import vietoris_rips_persistence, diagram_wasserstein_squared


def export_fixed_cloud_comparison(ph_run, baseline_run, output, seed, cloud_size=12,
                                 rare_anchors=3, medium_anchors=1, popular_anchors=1):
    """Compare initial/CE/PH item geometry using neighbors chosen once from CE.

    This diagnoses item embeddings; it is not the session/target loss itself.
    Anchor and neighbor selection uses TRAIN popularity only, never test hits.
    """
    ph_run, baseline_run, output = map(Path, (ph_run, baseline_run, output))
    if not 3 <= cloud_size <= 32:
        raise ValueError('Expected cloud_size in [3,32]')
    output.mkdir(parents=True, exist_ok=False)
    mapping = pd.read_csv(ph_run / 'item_embedding_mapping.csv').sort_values('internal_item_id').reset_index(drop=True)
    other = pd.read_csv(baseline_run / 'item_embedding_mapping.csv').sort_values('internal_item_id').reset_index(drop=True)
    if not mapping[['internal_item_id', 'item_id']].equals(other[['internal_item_id', 'item_id']]):
        raise ValueError('Checkpoint mappings differ')
    ce_info = json.loads((baseline_run / 'metrics.json').read_text(encoding='utf-8'))
    ph_info = json.loads((ph_run / 'metrics.json').read_text(encoding='utf-8'))
    if ce_info['experiment']['initial_embedding_sha256'] != ph_info['experiment']['initial_embedding_sha256']:
        raise ValueError('CE and PH were not initialized from identical embeddings')
    initial = np.load(ph_run / 'initial_item_embeddings.npz')['item_embeddings']
    arrays = {'initial': initial, 'ce': np.load(baseline_run / 'item_embeddings.npy'),
              'ph': np.load(ph_run / 'item_embeddings.npy')}
    expected_ids = np.arange(1, len(mapping) + 1)
    if not np.array_equal(mapping.internal_item_id.to_numpy(), expected_ids):
        raise ValueError('Expected row = internal ID minus one')
    if any(a.shape != initial.shape or len(a) != len(mapping) for a in arrays.values()):
        raise ValueError('Embedding dimensions differ')
    warm = np.flatnonzero(mapping.is_trainable_warm.to_numpy())
    if len(warm) < cloud_size:
        raise ValueError('Too few warm items')
    units = {name: F.normalize(torch.as_tensor(a, dtype=torch.float32), dim=1) for name, a in arrays.items()}
    rng = np.random.default_rng(seed)
    anchors = []
    for group, count in [('rare', rare_anchors), ('medium', medium_anchors), ('popular', popular_anchors)]:
        candidates = np.flatnonzero(mapping.is_trainable_warm.to_numpy() & mapping.popularity_group.eq(group).to_numpy())
        if len(candidates):
            anchors.extend((int(index), group) for index in rng.choice(candidates, min(count, len(candidates)), replace=False))
    records = []
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    for anchor, group in anchors:
        distances = (units['ce'][warm] - units['ce'][anchor]).square().sum(dim=1).numpy()
        candidates = warm[np.argsort(distances, kind='stable')]
        neighbors = np.r_[anchor, candidates[candidates != anchor][:cloud_size - 1]]
        assert len(neighbors) == cloud_size and len(np.unique(neighbors)) == cloud_size
        anchor_id = int(mapping.iloc[anchor].internal_item_id)
        diagrams = {name: vietoris_rips_persistence(unit[neighbors], max_dimension=1) for name, unit in units.items()}
        saved = {'neighbor_internal_ids': mapping.iloc[neighbors].internal_item_id.to_numpy(),
                 'neighbor_raw_item_ids': mapping.iloc[neighbors].item_id.to_numpy()}
        for name, diag in diagrams.items():
            saved[f'{name}_points'] = units[name][neighbors].numpy()
            for dimension in (0, 1):
                saved[f'{name}_H{dimension}'] = diag[dimension].numpy()
        np.savez_compressed(output / f'anchor_{anchor_id}.npz', **saved)
        fig, axes = plt.subplots(3, 2, figsize=(11, 11), layout='constrained')
        for row, (name, label) in enumerate([('initial', 'Before training'), ('ce', 'After CE'), ('ph', 'After PH H0+H1')]):
            left, right = axes[row]
            left.plot([0, 2], [0, 2], '--', color='#999999')
            feature = 0
            for dimension, color in [(0, '#3985AD'), (1, '#D68A20')]:
                d = diagrams[name][dimension].numpy()
                finite = d[np.isfinite(d).all(axis=1)]
                if len(finite):
                    left.scatter(finite[:, 0], finite[:, 1], s=25, color=color, label=f'H{dimension}')
                for birth, death in d:
                    right.hlines(feature, birth, death if np.isfinite(death) else 2.02,
                                 color=color, linestyle='-' if np.isfinite(death) else '--')
                    feature += 1
            left.set(title=f'{label}: persistence diagram', xlabel='Birth (chord distance)',
                     ylabel='Death (chord distance)', xlim=(-.05, 2.05), ylim=(-.05, 2.05))
            left.legend(loc='upper left')
            right.set(title=f'{label}: barcode', xlabel='Filtration distance', ylabel='Feature index', xlim=(-.05, 2.05))
        fig.suptitle(f'Fixed {cloud_size} TRAIN neighbor IDs; anchor internal ID {anchor_id}; group {group}', fontsize=13)
        fig.savefig(output / f'anchor_{anchor_id}.png', dpi=160)
        plt.close(fig)
        records.append({'anchor_internal_id': anchor_id, 'anchor_raw_id': int(mapping.iloc[anchor].item_id),
                        'group': group, 'cloud_size': cloud_size,
                        'ce_to_ph_H0_W2_squared': float(diagram_wasserstein_squared(diagrams['ce'][0], diagrams['ph'][0])),
                        'ce_to_ph_H1_W2_squared': float(diagram_wasserstein_squared(diagrams['ce'][1], diagrams['ph'][1])),
                        'ce_H1_bars': len(diagrams['ce'][1]), 'ph_H1_bars': len(diagrams['ph'][1])})
    pd.DataFrame(records).to_csv(output / 'anchor_manifest.csv', index=False)
    (output / 'provenance.json').write_text(json.dumps({
        'seed': seed, 'reference_for_neighbor_selection': str(baseline_run),
        'states': ['same_seed_initialization', 'best_CE_checkpoint', 'best_PH_checkpoint'],
        'neighbor_ids_fixed_across_states': True, 'anchor_selection': 'seeded TRAIN group sampling',
        'filtration': 'Vietoris-Rips on L2-normalized embeddings, Euclidean chord distance',
        'scope': 'item-geometry diagnostic; not the session-target training loss or proof of recommendation gain',
        'visualization_cloud_size': cloud_size,
    }, indent=2), encoding='utf-8')
    return records
