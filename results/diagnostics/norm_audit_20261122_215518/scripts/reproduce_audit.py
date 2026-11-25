"""Independent audit of frequency–norm measurements; no training or test tuning."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import sys
import hashlib

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr, rankdata
from torch.utils.data import DataLoader

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / 'src/btl_k03').is_dir())
sys.path.insert(0, str(ROOT / 'src'))
from btl_k03.models import build_model_from_config, SessionPrefixDataset
from btl_k03.utils import set_global_seed

torch.set_num_threads(2)
OUT = ROOT / 'results/diagnostics' / ('norm_audit_' + datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d_%H%M%S'))
OUT.mkdir(parents=True, exist_ok=False)
for subdir in ('tables', 'figures', 'data', 'predictions'):
    (OUT / subdir).mkdir()
print('OUTPUT', OUT, flush=True)
DATA = ROOT / 'data/processed/diginetica'
mapping = pd.read_csv(DATA / 'item_mapping.csv').sort_values('internal_item_id').reset_index(drop=True)
ids = mapping.internal_item_id.to_numpy().copy()
warm = mapping.is_trainable_warm.to_numpy().copy()
cold = mapping.is_controlled_cold.to_numpy().copy()
frequency = mapping.train_frequency.to_numpy().copy()
events = pd.read_parquet(DATA / 'train_events.parquet')
train = pd.read_parquet(DATA / 'train_examples.parquet')
assert np.array_equal(ids, np.arange(1, len(mapping) + 1))
event_counts = events.internal_item_id.value_counts().reindex(ids, fill_value=0).to_numpy()
target_counts = train.target_internal_item_id.value_counts().reindex(ids, fill_value=0).to_numpy()
prefix = np.concatenate([np.asarray(s[-50:], dtype=np.int64) for s in train.prefix_items])
input_counts = np.bincount(prefix, minlength=len(mapping) + 1)[ids]
data_check = {
    'mapping_contiguous_ids_1_to_N': True,
    'mapping_raw_ids_unique': bool(mapping.item_id.is_unique),
    'warm_items': int(warm.sum()), 'cold_items': int(cold.sum()),
    'train_events': len(events), 'train_examples': len(train),
    'frequency_mismatched_items': int(np.count_nonzero(event_counts != frequency)),
    'frequency_max_abs_error': int(np.abs(event_counts - frequency).max()),
    'train_event_sum_matches_frequency_sum': bool(event_counts.sum() == frequency.sum()),
    'warm_items_with_zero_frequency': int(np.count_nonzero(warm & (frequency <= 0))),
    'cold_items_with_positive_frequency': int(np.count_nonzero(cold & (frequency > 0))),
    'training_targets_cold_or_pad': int(np.count_nonzero(~np.isin(train.target_internal_item_id, ids[warm]))),
    'training_input_cold_or_pad': int(np.count_nonzero(~np.isin(prefix, ids[warm]))),
}
(OUT / 'data_checks.json').write_text(json.dumps(data_check, indent=2), encoding='utf-8')
print('DATA_CHECKS', data_check, flush=True)
del prefix, events
sample = train.sample(n=2048, random_state=1701).sort_index()
dataset = SessionPrefixDataset(sample, 50)
summary, group_rows, geometry_rows, gradient_rows, invariant_rows = [], [], [], [], []


def rho(a, b):
    return float(spearmanr(a, b).statistic)


for model_name in ('gru4rec', 'narm', 'sasrec'):
    for seed in (2024, 2025):
        run = ROOT / f'runs/{model_name}/seed_{seed}'
        checkpoint_path = run / 'checkpoint.pth'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state = checkpoint['model_state_dict']
        E = state['item_embedding.weight'].detach().cpu().numpy().astype(np.float64)
        item_E = E[ids]  # Direct ID indexing: no reliance on exported row order.
        norms = np.sqrt(np.sum(item_E * item_E, axis=1))
        exported = np.load(run / 'item_embeddings.npy', allow_pickle=False)
        saved_mapping = pd.read_csv(run / 'item_embedding_mapping.csv').sort_values('internal_item_id').reset_index(drop=True)
        table = pd.read_parquet(ROOT / f'results/item_level/{model_name}_seed{seed}.parquet')
        table = table.set_index('internal_item_id').loc[ids]
        rank_rho = float(np.corrcoef(rankdata(frequency[warm]), rankdata(norms[warm]))[0, 1])
        fw, nw = frequency[warm], norms[warm]
        center = item_E[warm].mean(axis=0)
        centered_norm = np.linalg.norm(item_E - center, axis=1)
        row = {
            'model': model_name, 'seed': seed, 'warm_items': int(warm.sum()),
            'rho_frequency_norm_warm': rho(fw, nw),
            'rho_frequency_norm_all_including_cold': rho(frequency, norms),
            'pearson_frequency_norm_warm': float(pearsonr(fw, nw).statistic),
            'pearson_log1p_frequency_norm_warm': float(pearsonr(np.log1p(fw), nw).statistic),
            'rho_target_count_norm_warm': rho(target_counts[warm], nw),
            'rho_input_count_norm_warm': rho(input_counts[warm], nw),
            'rho_frequency_recommendation_count': rho(fw, table.rec_count.to_numpy()[warm]),
            'rho_frequency_norm_centered_warm': rho(fw, centered_norm[warm]),
            'spearman_independent_rank_formula_abs_error': abs(rank_rho - rho(fw, nw)),
            'export_shape_correct': exported.shape == item_E.shape,
            'export_matches_checkpoint_exactly': bool(np.array_equal(exported, E[1:].astype(np.float32))),
            'checkpoint_mapping_matches_current': bool(saved_mapping[['internal_item_id', 'item_id']].equals(mapping[['internal_item_id', 'item_id']])),
            'saved_table_norm_max_abs_error': float(np.max(np.abs(table.embedding_norm.to_numpy() - norms))),
            'saved_table_frequency_mismatches': int(np.count_nonzero(table.train_freq.to_numpy() != frequency)),
            'saved_table_raw_id_mismatches': int(np.count_nonzero(table.item_id.to_numpy() != mapping.item_id.to_numpy())),
            'padding_norm': float(np.linalg.norm(E[0])),
            'nonfinite_embedding_values': int(np.count_nonzero(~np.isfinite(E))),
            'weight_decay': float(checkpoint['config']['training'].get('weight_decay', 0)),
            'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        }
        # Rebuild the original initialization, checking cold rows to determine
        # whether RNG/constructor reproduction is exact before using it.
        set_global_seed(seed)
        model = build_model_from_config(checkpoint['config'], mapping, torch.device('cpu'))
        init_E = model.item_embedding.weight.detach().numpy().copy()
        cold_exact = bool(np.array_equal(init_E[ids[cold]], E[ids[cold]].astype(np.float32)))
        row['cold_matches_reconstructed_initialization_exactly'] = cold_exact
        row['replica_initial_rho_frequency_norm_warm'] = rho(fw, np.linalg.norm(init_E[ids[warm]].astype(np.float64), axis=1))
        model.load_state_dict(state, strict=True)
        model.eval()
        with torch.no_grad():
            h_list = [model.encode_session(b['item_seq'], b['item_seq_len']) for b in DataLoader(dataset, batch_size=256)]
            H = torch.cat(h_list).numpy().astype(np.float64)
        mu_h = H.mean(axis=0)
        direction = mu_h / np.linalg.norm(mu_h)
        projection = item_E @ direction
        perpendicular_norm = np.sqrt(np.maximum(0, norms ** 2 - projection ** 2))
        row.update(rho_frequency_mean_session_projection=rho(fw, projection[warm]),
                   rho_frequency_perpendicular_norm=rho(fw, perpendicular_norm[warm]),
                   mean_session_direction_norm=float(np.linalg.norm(mu_h)))
        summary.append(row)
        for group in ('popular', 'medium', 'rare', 'cold'):
            selected = mapping.popularity_group.eq(group).to_numpy()
            group_rows.append({'model': model_name, 'seed': seed, 'group': group, 'items': int(selected.sum()),
                               'norm_mean': float(norms[selected].mean()), 'norm_median': float(np.median(norms[selected])),
                               'centered_norm_mean': float(centered_norm[selected].mean()),
                               'mean_session_projection': float(projection[selected].mean()),
                               'perpendicular_norm_mean': float(perpendicular_norm[selected].mean()),
                               'rho_frequency_norm_within_group': rho(frequency[selected], norms[selected]) if group != 'cold' else None})
        geometry_rows.append(pd.DataFrame({'model': model_name, 'seed': seed, 'internal_item_id': ids,
                                          'frequency': frequency, 'target_count': target_counts, 'input_count': input_counts,
                                          'group': mapping.popularity_group, 'norm': norms,
                                          'centered_norm': centered_norm, 'mean_session_projection': projection,
                                          'perpendicular_norm': perpendicular_norm}))
        # A single deterministic audit batch measures current gradients only;
        # it does not reconstruct the whole Adam training trajectory.
        if model_name == 'sasrec' and seed == 2024:
            batch = next(iter(DataLoader(dataset, batch_size=256)))
            sequence, lengths, targets = batch['item_seq'], batch['item_seq_len'], batch['target']
            weight = model.item_embedding.weight
            output_leaf = weight.detach().clone().requires_grad_(True)
            model.zero_grad(set_to_none=True)
            hidden = model.encode_session(sequence, lengths)
            logits = hidden @ output_leaf.T
            logits = logits.masked_fill(~model.trainable_warm_mask[None], -torch.inf)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            loss.backward()
            gi, go = weight.grad.detach().clone(), output_leaf.grad.detach().clone()
            model.zero_grad(set_to_none=True)
            tied = model.full_sort_logits(sequence, lengths)
            torch.nn.functional.cross_entropy(tied, targets).backward()
            total = weight.grad.detach()
            max_gradient_error = float((total - gi - go).abs().max())
            assert torch.allclose(total, gi + go, atol=2e-6, rtol=2e-4)
            unit = torch.nn.functional.normalize(weight.detach(), dim=1)
            for group in ('popular', 'medium', 'rare'):
                selected_ids = torch.as_tensor(ids[mapping.popularity_group.eq(group)])
                gradient_rows.append({'model': model_name, 'seed': seed, 'group': group,
                    'input_gradient_norm_sum': float(gi[selected_ids].norm(dim=1).sum()),
                    'output_gradient_norm_sum': float(go[selected_ids].norm(dim=1).sum()),
                    'mean_first_order_SGD_norm_change_output_per_unit_lr': float(-(go[selected_ids] * unit[selected_ids]).sum(dim=1).mean()),
                    'mean_first_order_SGD_norm_change_input_per_unit_lr': float(-(gi[selected_ids] * unit[selected_ids]).sum(dim=1).mean()),
                    'tied_gradient_reconstruction_max_abs_error': max_gradient_error,
                    'scope': 'single_eval_mode_train_batch; not Adam trajectory'})
            model.zero_grad(set_to_none=True)
            # SASRec symmetry: E_warm <- E_warm-c, P <- P+c. Valid input
            # vectors stay the same; warm logits change by one row-wise offset.
            # This is in-memory only. Padding queries are not used for ranking.
            with torch.no_grad():
                before_h = model.encode_session(sequence, lengths)
                before_logits = model.full_sort_logits(sequence, lengths)
                before_ids = before_logits.topk(20, dim=1).indices
                shift = weight[model.trainable_warm_mask].mean(dim=0).clone()
                weight[model.trainable_warm_mask] -= shift
                model.position_embedding.weight += shift
                after_h = model.encode_session(sequence, lengths)
                after_logits = model.full_sort_logits(sequence, lengths)
                after_ids = after_logits.topk(20, dim=1).indices
                live = model.trainable_warm_mask
                row_offset = (before_h @ shift)[:, None]
                residual = (after_logits[:, live] - before_logits[:, live] + row_offset).abs().max()
                invariant_rows.append({'model': model_name, 'seed': seed, 'queries': len(sequence),
                    'transformation': 'center warm E; add same center to positional embeddings',
                    'hidden_max_abs_error': float((after_h - before_h).abs().max()),
                    'row_shift_logit_residual_max_abs': float(residual),
                    'top20_identical_queries': int(before_ids.eq(after_ids).all(dim=1).sum()),
                    'top20_identical_fraction': float(before_ids.eq(after_ids).all(dim=1).float().mean()),
                    'original_rho': rho(fw, nw), 'translated_rho': rho(fw, centered_norm[warm]),
                    'checkpoint_modified': False})
            model.load_state_dict(state, strict=True)
        pd.DataFrame(summary).to_csv(OUT / 'tables' / 'checkpoint_checks_and_correlations.csv', index=False)
        pd.DataFrame(group_rows).to_csv(OUT / 'tables' / 'group_norm_statistics.csv', index=False)
        print('DONE', model_name, seed, 'rho=', row['rho_frequency_norm_warm'],
              'centered=', row['rho_frequency_norm_centered_warm'],
              'norm_error=', row['saved_table_norm_max_abs_error'], flush=True)
pd.concat(geometry_rows, ignore_index=True).to_parquet(OUT / 'data' / 'item_geometry.parquet', index=False)
pd.DataFrame(gradient_rows).to_csv(OUT / 'tables' / 'gradient_diagnostic.csv', index=False)
pd.DataFrame(invariant_rows).to_csv(OUT / 'tables' / 'sasrec_translation_invariance.csv', index=False)
checks = pd.DataFrame(summary)
failures = {
    'data_frequency_error': data_check['frequency_mismatched_items'],
    'id_mapping_failures': int((~checks.checkpoint_mapping_matches_current).sum()),
    'embedding_export_failures': int((~checks.export_matches_checkpoint_exactly).sum()),
    'norm_table_failures_over_1e-6': int((checks.saved_table_norm_max_abs_error > 1e-6).sum()),
    'frequency_table_mismatches': int(checks.saved_table_frequency_mismatches.sum()),
    'raw_id_table_mismatches': int(checks.saved_table_raw_id_mismatches.sum()),
    'nonfinite_embedding_values': int(checks.nonfinite_embedding_values.sum()),
}
(OUT / 'audit_verdict.json').write_text(json.dumps({'failures': failures,
    'all_six_warm_correlations_negative': bool(checks.rho_frequency_norm_warm.lt(0).all()),
    'no_checkpoint_files_modified': True,
    'mechanistic_scope': 'cross-sectional measurements, algebraic invariance and one gradient batch; not training-trajectory causality'}, indent=2), encoding='utf-8')
print('CHECKS\n', checks[['model', 'seed', 'rho_frequency_norm_warm', 'rho_frequency_norm_all_including_cold',
      'rho_target_count_norm_warm', 'rho_frequency_norm_centered_warm', 'rho_frequency_mean_session_projection']].to_string(index=False), flush=True)
print('INVARIANCE', invariant_rows, flush=True)
print('FAILURES', failures, flush=True)
print('COMPLETE', OUT, flush=True)
