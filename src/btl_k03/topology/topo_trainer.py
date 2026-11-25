"""Reproducible CE/cosine/PH controls using the current SBR model APIs."""
from __future__ import annotations

import copy
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..models import SessionPrefixDataset, build_model_from_config, build_top20_dataframe, predict_topk_for_dataset
from ..paths import PROCESSED_DIGI_DIR, RUNS_DIR
from ..popularity_metrics import compute_stratified_ranking_metrics, evaluate_top20_popularity_bias
from ..utils import create_run_logger, load_experiment_config, load_item_mapping, save_json, save_yaml, set_global_seed
from .persistence_loss import CosineTailAlignmentLoss, TopologicalPersistenceLoss, session_dispersion
from .role_gradients import ROLE_METHODS, backward_role_loss

METHODS = ("baseline", "dispersion", "cosine", "ph_h0", "ph_h0_h1", *ROLE_METHODS)


def _sample(frame, limit, seed):
    if limit is not None and limit < 1:
        raise ValueError("Sample limits must be positive or None")
    if limit is not None and limit < len(frame):
        frame = frame.sample(n=limit, random_state=seed).sort_index()
    # Retain original row index so prediction pairs are traceable across runs.
    return frame.copy()


def _evaluate(model, frame, mapping, device, batch_size, topk, output, split):
    dataset = SessionPrefixDataset(frame, max_seq_length=model._experiment_max_seq_length)
    predictions, scores = predict_topk_for_dataset(model, dataset, device, batch_size, topk)
    if not model.trainable_warm_mask[torch.as_tensor(predictions, device=device)].all():
        raise AssertionError("Cold/padding item leaked into recommendations")
    stratified = compute_stratified_ranking_metrics(predictions, dataset.targets, dataset.target_groups, k=topk)
    records = stratified.to_dict(orient="records")
    for row in records:
        if row["num_examples"] == 0:
            for metric in ("recall", "mrr", "ndcg"):
                row[f"{metric}@{topk}"] = None
    hits = predictions == dataset.targets[:, None]
    hit = hits.any(axis=1)
    ranks = hits.argmax(axis=1) + 1
    per_query = pd.DataFrame({"source_row": frame.index.to_numpy(), "session_id": dataset.session_ids,
                              "target": dataset.targets, "group": dataset.target_groups,
                              "hit": hit.astype(int), "reciprocal_rank": np.where(hit, 1 / ranks, 0),
                              "ndcg": np.where(hit, 1 / np.log2(ranks + 1), 0)})
    per_query.to_csv(output / f"{split}_per_query.csv.gz", index=False)
    lookup = np.zeros(int(mapping.internal_item_id.max()) + 1, dtype=np.int64)
    lookup[mapping.internal_item_id.to_numpy()] = mapping.item_id.to_numpy()
    top_frame = build_top20_dataframe(dataset, predictions, scores, lookup)
    top_frame.to_parquet(output / f"{split}_top20.parquet", index=False)
    bias, _, exposure = evaluate_top20_popularity_bias(top_frame, mapping, k=topk)
    return {"ranking_by_group": records, "popularity_bias": bias,
            "exposure_by_group": exposure.to_dict(orient="records")}


def train_and_evaluate_topology_seed(
    seed: int, config: dict | None = None, processed_dir: Path = PROCESSED_DIGI_DIR,
    max_epochs_override: int | None = None, device_str: str | None = None,
    *, model_name: str = "sasrec", method: str = "ph_h0_h1", output_dir: Path | None = None,
    train_limit: int | None = None, validation_limit: int | None = None,
    evaluate_test: bool = False, init_checkpoint: Path | None = None,
) -> dict:
    """Train in a fresh directory; select on validation MRR, test only if asked.

    Limited runs are exploratory pilots, never comparable to full-data baselines.
    An optional common checkpoint permits a controlled fine-tuning pilot. Adam
    state is deliberately fresh for every method; this is not training resume.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    cfg = copy.deepcopy(config) if config is not None else load_experiment_config(model_name)
    model_name = cfg.get("model_name", model_name).lower()
    cfg.setdefault("topology_params", {})
    cfg.update(seed=int(seed), method=method)
    if max_epochs_override is not None:
        cfg["training"]["epochs"] = int(max_epochs_override)
    training, topology = cfg["training"], cfg["topology_params"]
    epochs = int(training.get("epochs", 30))
    interval = int(topology.get("regularization_interval", 1))
    if interval < 1:
        raise ValueError("regularization_interval must be positive")
    if epochs < 1 or float(topology.get("lambda_topo", 0.15)) < 0:
        raise ValueError("epochs must be positive and lambda_topo non-negative")
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_global_seed(seed)
    output = Path(output_dir) if output_dir is not None else RUNS_DIR / "persistent_homology" / model_name / method / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=False)
    logger = create_run_logger(output / "training.log", f"ph.{model_name}.{method}.{seed}")
    mapping = load_item_mapping(processed_dir)
    train = _sample(pd.read_parquet(processed_dir / cfg["data_files"]["train_examples"]), train_limit, seed)
    validation = _sample(pd.read_parquet(processed_dir / cfg["data_files"]["validation_examples"]), validation_limit, seed + 1)
    if train.empty or validation.empty:
        raise ValueError("Training and validation data must be nonempty")
    model = build_model_from_config(cfg, mapping, device)
    if init_checkpoint is not None:
        checkpoint = torch.load(init_checkpoint, map_location=device, weights_only=False)
        if checkpoint.get("config", {}).get("model_name") != model_name:
            raise ValueError("Initial checkpoint architecture does not match")
        checkpoint_mapping = pd.read_csv(init_checkpoint.parent / "item_embedding_mapping.csv").sort_values("internal_item_id")
        if not mapping[["internal_item_id", "item_id"]].reset_index(drop=True).equals(
            checkpoint_mapping[["internal_item_id", "item_id"]].reset_index(drop=True)
        ):
            raise ValueError("Initial checkpoint item IDs differ from current mapping")
        model.load_state_dict(checkpoint["model_state_dict"])
        expected_mask = torch.zeros_like(model.trainable_warm_mask)
        expected_mask[torch.tensor(mapping.loc[mapping.is_trainable_warm, "internal_item_id"].to_numpy(), device=device)] = True
        if not torch.equal(model.trainable_warm_mask, expected_mask):
            raise ValueError("Initial checkpoint warm mask differs from current mapping")
    model._experiment_max_seq_length = int(cfg.get("max_seq_length", 50))
    for frame in (train, validation):
        ids = torch.tensor(frame.target_internal_item_id.to_numpy(), device=device)
        if not model.trainable_warm_mask[ids].all():
            raise ValueError("Non-warm training/validation target")
    frozen = ~model.trainable_warm_mask
    initial_frozen = model.item_embedding.weight.detach()[frozen].clone()
    frequencies = torch.zeros(model.num_items_with_pad, dtype=torch.long, device=device)
    frequencies[torch.tensor(mapping.internal_item_id.to_numpy(), device=device)] = torch.tensor(mapping.train_frequency.to_numpy(), device=device)
    if method == "cosine":
        criterion = CosineTailAlignmentLoss(tau=float(topology.get("cosine_tau", 0.45)),
            lambda_anisotropy=float(topology.get("lambda_anisotropy", 0.05)), tail_threshold=int(topology.get("tail_threshold", 20)))
    else:
        criterion = TopologicalPersistenceLoss(neighbors=int(topology.get("neighbors", 8)),
            max_clouds=int(topology.get("max_clouds", 8)), max_dimension=0 if method == "ph_h0" else 1,
            h1_weight=float(topology.get("h1_weight", 1)), lambda_anisotropy=float(topology.get("lambda_anisotropy", 0.05)),
            tail_threshold=int(topology.get("tail_threshold", 20)))
        criterion.sampling_generator.manual_seed(seed)
    batch_size = int(training.get("train_batch_size", 512))
    eval_batch_size = int(training.get("eval_batch_size", 1024))
    topk = int(cfg["evaluation"].get("topk", 20))
    if not 1 <= topk <= int(model.trainable_warm_mask.sum()):
        raise ValueError("topk exceeds the warm candidate set")
    loader = DataLoader(SessionPrefixDataset(train, model._experiment_max_seq_length), batch_size=batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training.get("learning_rate", 0.001)),
                                 weight_decay=float(training.get("weight_decay", 0)))
    # PH pairing/assignment runs in FP32. No implicit mixed-precision changes
    # between methods, especially for small persistence intervals.
    cfg["training"]["use_amp"] = False
    cfg["experiment"] = {"train_examples": len(train), "validation_examples": len(validation),
        "train_limit": train_limit, "validation_limit": validation_limit,
        "initial_checkpoint": str(init_checkpoint.resolve()) if init_checkpoint else None,
        "evaluate_test": evaluate_test, "reference": "current_batch_target_embeddings_stop_gradient",
        "filtration": "vietoris_rips_euclidean_chord", "coefficient_field": 2,
        "precision": "float32", "device": str(device)}
    initial_embedding = model.item_embedding.weight.detach().cpu().numpy().copy()
    cfg["experiment"]["initial_embedding_sha256"] = hashlib.sha256(initial_embedding.tobytes()).hexdigest()
    if topology.get("save_initial_embeddings", False) and method == "ph_h0_h1":
        np.savez_compressed(output / "initial_item_embeddings.npz", item_embeddings=initial_embedding[1:])
    del initial_embedding
    save_yaml(cfg, output / "config.yaml")
    pd.DataFrame({"source_row": train.index, "session_id": train.session_id.to_numpy()}).to_csv(output / "train_sample.csv.gz", index=False)
    best, best_epoch, stale = -float("inf"), 0, 0
    history = []
    start = time.perf_counter()
    logger.info("%s %s seed=%s train=%s validation=%s device=%s", model_name, method, seed, len(train), len(validation), device)
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        totals = {"ce": 0.0, "regularizer": 0.0, "ph_h0": 0.0, "ph_h1": 0.0,
                  "ph_clouds": 0.0, "ph_h1_bars": 0.0, "adjusted_items": 0.0,
                  "input_active_items": 0.0, "role_conflict_items": 0.0,
                  "gradient_adjustment_norm": 0.0, "relative_gradient_change": 0.0}
        samples = 0
        for batch_index, batch in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            targets = batch["target"].to(device)
            session = model.encode_session(batch["item_seq"].to(device), batch["item_seq_len"].to(device))
            output_weight = model.item_embedding.weight.detach().requires_grad_() if method in ROLE_METHODS else model.item_embedding.weight
            logits = (session @ output_weight.T).masked_fill(~model.trainable_warm_mask[None], -1e9)
            ce = F.cross_entropy(logits, targets)
            apply_regularizer = batch_index % interval == 0
            if method == "baseline" or (method in ROLE_METHODS and method != "role_ph") or not apply_regularizer:
                regularizer = session.sum() * 0
            elif method == "dispersion":
                regularizer = float(topology.get("lambda_anisotropy", 0.05)) * session_dispersion(session)
            else:
                regularizer = criterion(session, model.item_embedding(targets), frequencies, targets)
            # Periodic regularization saves exact-PH pairing work on GPU. The
            # multiplier preserves the nominal per-step strength in expectation
            # over shuffled batches; it is not identical to every-step training.
            loss = ce if method in ROLE_METHODS else ce + float(topology.get("lambda_topo", 0.15)) * interval * regularizer
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            if method in ROLE_METHODS:
                effective_method = "role_sum" if method == "role_ph" and not apply_regularizer else method
                role_stats = backward_role_loss(ce, model.item_embedding.weight, output_weight,
                    model.trainable_warm_mask, frequencies, effective_method,
                    strength=float(topology.get("gradient_strength", .5)),
                    tail_threshold=int(topology.get("tail_threshold", 20)), topology_loss=regularizer)
                for key in ("adjusted_items", "input_active_items", "role_conflict_items", "gradient_adjustment_norm", "relative_gradient_change"):
                    totals[key] += role_stats[key]
            else:
                loss.backward()
            clip = float(training.get("grad_clip_norm", 5))
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
            optimizer.step()
            # Gradient masking alone does not protect against weight decay.
            with torch.no_grad():
                model.item_embedding.weight[frozen] = initial_frozen
            count = len(targets)
            totals["ce"] += float(ce.detach()) * count
            totals["regularizer"] += float(regularizer.detach()) * count
            if (method.startswith("ph_") or method == "role_ph") and apply_regularizer:
                for key in ("ph_h0", "ph_h1"):
                    totals[key] += criterion.last_stats[key] * count
                for key in ("ph_clouds", "ph_h1_bars"):
                    totals[key] += criterion.last_stats[key]
            samples += count
            log_interval = int(training.get("log_every_batches", 0))
            if log_interval > 0 and ((batch_index + 1) % log_interval == 0 or batch_index + 1 == len(loader)):
                logger.info("epoch=%s batch=%s/%s CE=%.5f regularizer=%.5f", epoch,
                            batch_index + 1, len(loader), float(ce.detach()), float(regularizer.detach()))
        val_dataset = SessionPrefixDataset(validation, model._experiment_max_seq_length)
        predicted, _ = predict_topk_for_dataset(model, val_dataset, device, eval_batch_size, topk)
        metrics = compute_stratified_ranking_metrics(predicted, val_dataset.targets, val_dataset.target_groups, k=topk)
        score = float(metrics.iloc[0][f"mrr@{topk}"])
        if score > best + float(cfg["early_stopping"].get("min_delta", 0.0001)):
            best, best_epoch, stale = score, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "config": cfg, "epoch": epoch,
                        "best_validation_mrr": best}, output / "checkpoint.pth")
        else:
            stale += 1
        row = {"epoch": epoch, **{k: v / samples if k in ("ce", "regularizer", "ph_h0", "ph_h1") else v for k, v in totals.items()},
               "validation_mrr": score, "validation_recall": float(metrics.iloc[0][f"recall@{topk}"]),
               "seconds": time.perf_counter() - epoch_start}
        history.append(row)
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        logger.info("epoch=%s CE=%.5f regularizer=%.5f PH0=%.5f PH1=%.5f clouds=%s H1_bars=%s val_MRR=%.5f time=%.1fs",
                    epoch, row["ce"], row["regularizer"], row["ph_h0"], row["ph_h1"],
                    row["ph_clouds"], row["ph_h1_bars"], score, row["seconds"])
        if stale >= int(cfg["early_stopping"].get("patience", 5)):
            break
    model.load_state_dict(torch.load(output / "checkpoint.pth", map_location=device, weights_only=False)["model_state_dict"])
    drift = float((model.item_embedding.weight.detach()[frozen] - initial_frozen).abs().max())
    if drift != 0:
        raise AssertionError(f"Frozen embedding drift: {drift}")
    result = {"model": model_name, "method": method, "seed": seed, "best_epoch": best_epoch,
              "best_validation_mrr": best,
              "frozen_embedding_max_abs_drift": drift, "seconds": time.perf_counter() - start,
              "status": "exploratory_pilot" if train_limit or validation_limit or init_checkpoint else "full_training",
              "experiment": cfg["experiment"], "output_dir": str(output.resolve())}
    result["validation"] = _evaluate(model, validation, mapping, device, eval_batch_size, topk, output, "validation")
    # Load test data only after selecting the checkpoint and only when requested.
    if evaluate_test:
        test = pd.read_parquet(processed_dir / cfg["data_files"]["test_examples"])
        result["test"] = _evaluate(model, test, mapping, device, eval_batch_size, topk, output, "test")
    np.save(output / "item_embeddings.npy", model.item_embedding.weight.detach().cpu().numpy()[1:])
    mapping.to_csv(output / "item_embedding_mapping.csv", index=False)
    result["training_and_selection_seconds"] = result["seconds"]
    result["seconds"] = time.perf_counter() - start
    save_json(result, output / "metrics.json")
    return result
