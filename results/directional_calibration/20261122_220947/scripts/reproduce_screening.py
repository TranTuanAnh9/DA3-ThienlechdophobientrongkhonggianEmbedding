"""Exploratory validation screening of mean-session-direction corrections."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / 'src/btl_k03').is_dir())
sys.path.insert(0, str(ROOT / 'src'))
from btl_k03.models import SessionPrefixDataset, build_model_from_config
from btl_k03.popularity_metrics import compute_gini_coefficient, compute_ranking_metrics_from_predictions
from btl_k03.statistics import clustered_delta

torch.set_num_threads(2)
CFG = {
    'seeds': [2024, 2025], 'model': 'sasrec',
    'direction_coefficients': [0., .025, .05, .10, .20, .40, .60, 1.],
    'popularity_coefficients': [0., .025, .05, .10, .15, .20, .30, .40],
    'methods': ['direction', 'constant_direction', 'matched_random', 'popularity'],
    'train_direction_sample': 16384, 'train_direction_sample_seed': 1701,
    'validation_session_split_seed': 2024, 'selection_session_fraction': .5,
    'max_overall_recall_drop_pp': .2, 'max_overall_mrr_drop_pp': .1,
    'require_rare_recall_increase_each_seed': True,
    'require_rare_mrr_nondecreasing_each_seed': True,
    'bootstrap_iterations': 2000, 'batch_size': 256, 'topk': 20,
    'test_evaluated': False, 'retrained': False,
    'scope': 'exploratory internal validation split; backbone checkpoint previously selected using full validation',
}
OUT = ROOT / 'results/directional_calibration' / datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d_%H%M%S')
OUT.mkdir(parents=True, exist_ok=False)
for subdir in ('tables', 'figures', 'data', 'predictions'):
    (OUT / subdir).mkdir()
(OUT / 'locked_protocol.json').write_text(json.dumps(CFG, indent=2), encoding='utf-8')
print('OUTPUT', OUT, flush=True)
DATA = ROOT / 'data/processed/diginetica'
mapping = pd.read_csv(DATA / 'item_mapping.csv').sort_values('internal_item_id').reset_index(drop=True)
ids = mapping.internal_item_id.to_numpy().copy()
warm_ids = ids[mapping.is_trainable_warm.to_numpy()]
frequency = torch.zeros(int(ids.max()) + 1)
frequency[torch.tensor(ids)] = torch.tensor(mapping.train_frequency.to_numpy().copy(), dtype=torch.float32)
train = pd.read_parquet(DATA / 'train_examples.parquet')
train_sample = train.sample(CFG['train_direction_sample'], random_state=CFG['train_direction_sample_seed']).sort_index()
validation = pd.read_parquet(DATA / 'validation_examples.parquet')
sessions = np.sort(validation.session_id.unique())
rng = np.random.default_rng(CFG['validation_session_split_seed'])
rng.shuffle(sessions)
selection_sessions = sessions[:len(sessions) // 2]
selection_mask = validation.session_id.isin(selection_sessions).to_numpy()
confirm_mask = ~selection_mask
assert not set(validation.loc[selection_mask, 'session_id']) & set(validation.loc[confirm_mask, 'session_id'])
pd.DataFrame({'session_id': sessions, 'subset': np.where(np.isin(sessions, selection_sessions), 'selection', 'confirmation')}).to_csv(OUT / 'tables' / 'validation_session_split.csv', index=False)
print('VALIDATION_SPLIT', int(selection_mask.sum()), 'selection queries;', int(confirm_mask.sum()), 'confirmation queries', flush=True)


def fingerprint(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def metrics(top_ids, target, group):
    rare = group == 'rare'
    rank = compute_ranking_metrics_from_predictions(top_ids, target, 20)
    rare_rank = compute_ranking_metrics_from_predictions(top_ids[rare], target[rare], 20)
    counts = np.bincount(top_ids.ravel(), minlength=len(frequency))
    return {**rank, **{'rare_' + k: v for k, v in rare_rank.items()},
            'coverage@20': float(np.count_nonzero(counts[warm_ids]) / len(warm_ids)),
            'gini@20': compute_gini_coefficient(counts[warm_ids]),
            'arp@20': float(frequency.numpy()[top_ids].mean()),
            'n_queries': len(target), 'rare_queries': int(rare.sum())}


def queries(top_ids, data):
    hits = top_ids == data.target_internal_item_id.to_numpy()[:, None]
    hit = hits.any(axis=1)
    ranks = hits.argmax(axis=1) + 1
    return pd.DataFrame({'source_row': data.index.to_numpy(), 'session_id': data.session_id.to_numpy(),
                         'target': data.target_internal_item_id.to_numpy(),
                         'group': data.target_popularity_group.to_numpy(),
                         'hit': hit.astype(int), 'reciprocal_rank': np.where(hit, 1 / ranks, 0)})


# Verify score correction equals steering the session vector in a tiny case.
test_h = torch.tensor([[1., 2.], [-2., 3.]])
test_e = torch.tensor([[2., 1.], [1., -1.], [0., 3.]])
test_u = torch.tensor([.6, .8])
for coefficient in (0., .2, 1.):
    a = test_h @ test_e.T - coefficient * (test_h @ test_u)[:, None] * (test_e @ test_u)[None]
    b = (test_h - coefficient * (test_h @ test_u)[:, None] * test_u[None]) @ test_e.T
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
print('ALGEBRA_CHECK_PASSED', flush=True)

variants = [('baseline', 0.)]
for method in CFG['methods']:
    grid = CFG['popularity_coefficients'] if method == 'popularity' else CFG['direction_coefficients']
    variants.extend((method, float(coefficient)) for coefficient in grid if coefficient != 0)
selection_rows, confirmations, saved_predictions = [], [], {}
started = time.perf_counter()
for seed in CFG['seeds']:
    folder = OUT / 'predictions' / f'seed_{seed}'; folder.mkdir()
    run = ROOT / f'runs/sasrec/seed_{seed}'
    checkpoint = torch.load(run / 'checkpoint.pth', map_location='cpu', weights_only=False)
    saved_map = pd.read_csv(run / 'item_embedding_mapping.csv').sort_values('internal_item_id').reset_index(drop=True)
    assert saved_map[['internal_item_id', 'item_id']].equals(mapping[['internal_item_id', 'item_id']])
    model = build_model_from_config(checkpoint['config'], mapping, torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    initial_fingerprint = fingerprint(model)
    train_dataset = SessionPrefixDataset(train_sample, int(checkpoint['config'].get('max_seq_length', 50)))
    dataset = SessionPrefixDataset(validation, train_dataset.max_seq_length)
    with torch.inference_mode():
        hidden = torch.cat([model.encode_session(b['item_seq'], b['item_seq_len']) for b in DataLoader(train_dataset, batch_size=256)])
        mean_h = hidden.mean(dim=0)
        u = torch.nn.functional.normalize(mean_h, dim=0)
        random_direction = torch.randn(u.shape, generator=torch.Generator().manual_seed(2026))
        random_direction = torch.nn.functional.normalize(random_direction, dim=0)
        E = model.item_embedding.weight.detach()
        projection = E @ u
        random_projection = E @ random_direction
        # Centering item projections only adds a row-wise constant to scores.
        projection -= projection[warm_ids].mean()
        random_projection -= random_projection[warm_ids].mean()
        train_scalar = hidden @ u
        random_scalar = hidden @ random_direction
        rms = torch.sqrt(train_scalar.square().mean() * projection[warm_ids].square().mean())
        random_rms = torch.sqrt(random_scalar.square().mean() * random_projection[warm_ids].square().mean())
        random_scale = float(rms / random_rms.clamp_min(1e-12))
        constant_scalar = float(mean_h @ u)
        penalty = torch.log1p(frequency)
        predictions = {v: np.empty((len(dataset), 20), dtype=np.int32) for v in variants}
        loader = DataLoader(dataset, batch_size=CFG['batch_size'], shuffle=False)
        offset = 0
        for index, batch in enumerate(loader):
            h = model.encode_session(batch['item_seq'], batch['item_seq_len'])
            raw = h @ E.T
            direction_correction = (h @ u)[:, None] * projection[None]
            random_correction = random_scale * (h @ random_direction)[:, None] * random_projection[None]
            constant_correction = constant_scalar * projection[None]
            for method, coefficient in variants:
                if method == 'baseline': adjusted = raw
                elif method == 'direction': adjusted = raw - coefficient * direction_correction
                elif method == 'constant_direction': adjusted = raw - coefficient * constant_correction
                elif method == 'matched_random': adjusted = raw - coefficient * random_correction
                else: adjusted = raw - coefficient * penalty[None]
                masked = adjusted.masked_fill(~model.trainable_warm_mask[None], -torch.inf)
                top_ids = masked.topk(20, dim=1).indices
                assert model.trainable_warm_mask[top_ids].all()
                predictions[(method, coefficient)][offset:offset + len(h)] = top_ids.numpy()
            if index == 0:
                original_top = model.full_sort_logits(batch['item_seq'], batch['item_seq_len']).topk(20, dim=1).indices.numpy()
                assert np.array_equal(predictions[('baseline', 0.)][:len(h)], original_top)
            offset += len(h)
            if index % 10 == 0 or offset == len(dataset):
                print(f'seed={seed} validation={offset}/{len(dataset)} variants={len(variants)} elapsed={time.perf_counter()-started:.1f}s', flush=True)
    assert fingerprint(model) == initial_fingerprint
    np.savez_compressed(folder / 'train_estimated_directions.npz', mean_session_direction=u.numpy(),
                        mean_train_session=mean_h.numpy(), random_direction=random_direction.numpy(),
                        train_source_rows=train_sample.index.to_numpy())
    (folder / 'verification.json').write_text(json.dumps({'parameters_unchanged': True,
        'state_sha256': initial_fingerprint, 'baseline_matches_original': True,
        'train_sample': len(train_sample), 'random_penalty_train_rms_match_scale': random_scale,
        'direction_source': 'train prefixes only', 'test_evaluated': False}, indent=2), encoding='utf-8')
    for method, coefficient in variants:
        ids_for_variant = predictions[(method, coefficient)]
        selection_rows.append({'seed': seed, 'method': method, 'coefficient': coefficient,
            **metrics(ids_for_variant[selection_mask], dataset.targets[selection_mask], dataset.target_groups[selection_mask])})
    # Confirmation labels/metrics are not consulted until policies are locked.
    saved_predictions[seed] = predictions
    pd.DataFrame(selection_rows).to_csv(OUT / 'tables' / 'selection_grid.csv', index=False)

selection = pd.DataFrame(selection_rows)
baseline = selection.loc[selection.method.eq('baseline')].set_index('seed')
policies, candidates = [], []
for method in CFG['methods']:
    eligible = []
    for coefficient, rows in selection.loc[selection.method.eq(method)].groupby('coefficient'):
        rows = rows.set_index('seed').loc[CFG['seeds']]
        delta = rows.select_dtypes('number') - baseline.select_dtypes('number')
        good = ((delta['recall@20'] >= -CFG['max_overall_recall_drop_pp'] / 100)
                & (delta['mrr@20'] >= -CFG['max_overall_mrr_drop_pp'] / 100)
                & (delta['rare_recall@20'] > 0) & (delta['rare_mrr@20'] >= 0))
        result = {'method': method, 'coefficient': float(coefficient), 'passes_both_selection_seeds': bool(good.all()),
                  'selection_rare_recall_delta_pp': float(delta['rare_recall@20'].mean() * 100)}
        candidates.append(result)
        if good.all(): eligible.append(result)
    winner = max(eligible, key=lambda r: r['selection_rare_recall_delta_pp']) if eligible else None
    policies.append({'method': method, 'coefficient': winner['coefficient'] if winner else 0.,
                     'nonzero_selection_passed': winner is not None})
pd.DataFrame(candidates).to_csv(OUT / 'tables' / 'selection_constraints.csv', index=False)
(OUT / 'locked_policies.json').write_text(json.dumps({'selected_on': 'selection sessions only',
    'confirmation_seen_during_selection': False, 'policies': policies}, indent=2), encoding='utf-8')

bootstrap_rows = []
for seed in CFG['seeds']:
    predictions = saved_predictions[seed]
    held = validation.loc[confirm_mask]
    base_query = queries(predictions[('baseline', 0.)][confirm_mask], held)
    base_query.to_csv(OUT / 'predictions' / f'seed_{seed}/baseline_confirmation_per_query.csv.gz', index=False)
    confirmations.append({'seed': seed, 'method': 'baseline', 'coefficient': 0.,
        **metrics(predictions[('baseline', 0.)][confirm_mask], held.target_internal_item_id.to_numpy(), held.target_popularity_group.to_numpy())})
    selected_queries = {'baseline': base_query}
    for policy in policies:
        method, coefficient = policy['method'], policy['coefficient']
        key = (method, coefficient) if coefficient else ('baseline', 0.)
        top_ids = predictions[key][confirm_mask]
        query = queries(top_ids, held)
        selected_queries[method] = query
        query.to_csv(OUT / 'predictions' / f'seed_{seed}/{method}_confirmation_per_query.csv.gz', index=False)
        confirmations.append({'seed': seed, 'method': method, 'coefficient': coefficient,
            **metrics(top_ids, held.target_internal_item_id.to_numpy(), held.target_popularity_group.to_numpy())})
        for group, metric in [('rare', 'hit'), ('rare', 'reciprocal_rank'), ('all_warm', 'hit'), ('all_warm', 'reciprocal_rank')]:
            bootstrap_rows.append({'seed': seed, 'treatment': method, 'control': 'baseline', 'group': group, 'metric': metric,
                **clustered_delta(base_query, query, metric, group, CFG['bootstrap_iterations'], np.random.default_rng(seed))})
    for control in ('popularity', 'constant_direction', 'matched_random'):
        for group, metric in [('rare', 'hit'), ('rare', 'reciprocal_rank'), ('all_warm', 'hit')]:
            bootstrap_rows.append({'seed': seed, 'treatment': 'direction', 'control': control, 'group': group, 'metric': metric,
                **clustered_delta(selected_queries[control], selected_queries['direction'], metric, group,
                                  CFG['bootstrap_iterations'], np.random.default_rng(seed))})
confirmation = pd.DataFrame(confirmations)
confirmation.to_csv(OUT / 'tables' / 'confirmation_comparison.csv', index=False)
confirmation.groupby('method').mean(numeric_only=True).to_csv(OUT / 'tables' / 'confirmation_means.csv')
pd.DataFrame(bootstrap_rows).to_csv(OUT / 'tables' / 'confirmation_bootstrap.csv', index=False)
summary = {'checkpoint_frozen': True, 'test_evaluated': False, 'policies': policies,
    'validation_selection_queries': int(selection_mask.sum()), 'validation_confirmation_queries': int(confirm_mask.sum()),
    'confirmation_rare_queries': int(validation.loc[confirm_mask].target_popularity_group.eq('rare').sum()),
    'seconds': time.perf_counter() - started, 'scope': CFG['scope'],
    'hypothesis': 'remove some score contribution along mean train-session direction; not a proven semantic/popularity disentanglement'}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print('LOCKED_POLICIES', policies, flush=True)
print('CONFIRMATION_MEANS\n', confirmation.groupby('method')[['recall@20', 'mrr@20', 'rare_recall@20', 'rare_mrr@20', 'coverage@20', 'gini@20']].mean().to_string(), flush=True)
print('COMPLETE', OUT, flush=True)
