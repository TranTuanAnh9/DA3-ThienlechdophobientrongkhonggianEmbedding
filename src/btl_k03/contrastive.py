"""SASRec controls with uniform or popularity-aware item dropout.

Dropped items are compacted, preserving chronological order and the most recent
event. This is item deletion, not a learned MASK token. No improvement/novelty
claim is implied. Fit and select on train/validation; read test only at the end.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .models import SessionPrefixDataset, build_model_from_config, predict_topk_for_dataset
from .paths import PROCESSED_DIGI_DIR
from .popularity_metrics import compute_ranking_metrics_from_predictions
from .topology.topo_trainer import _evaluate
from .utils import create_run_logger, load_experiment_config, load_item_mapping, save_json, save_yaml, set_global_seed

METHODS = ("baseline", "uniform", "head")


def generate_popularity_aware_views(item_seq, item_seq_len, head_mask,
                                    mask_prob_head=.4, mask_prob_other=.1, generator=None):
    if item_seq.ndim != 2 or item_seq_len.shape != (len(item_seq),):
        raise ValueError("Expected sequences [B,L] and lengths [B]")
    if not 0 <= mask_prob_head <= 1 or not 0 <= mask_prob_other <= 1:
        raise ValueError("Mask probabilities must be in [0,1]")
    if (item_seq_len < 1).any() or (item_seq_len > item_seq.shape[1]).any():
        raise ValueError("Session lengths must be 1..L")
    positions = torch.arange(item_seq.shape[1], device=item_seq.device)[None]
    valid = (positions < item_seq_len[:, None]) & item_seq.ne(0)
    probability = torch.where(head_mask[item_seq].bool(), mask_prob_head, mask_prob_other)
    views = []
    for _ in range(2):
        keep = valid & (torch.rand(item_seq.shape, device=item_seq.device, generator=generator) >= probability)
        # Do not turn a short prefix into an empty or unrelated context.
        keep.scatter_(1, (item_seq_len - 1)[:, None], True)
        new_lengths = keep.sum(dim=1)
        destination = (keep.long().cumsum(1) - 1).clamp_min(0)
        compact = torch.zeros_like(item_seq)
        compact.scatter_add_(1, destination, item_seq * keep)
        views.extend((compact, new_lengths))
    return tuple(views)


def compute_infonce_contrastive_loss(repr_1, repr_2, temperature=.1, session_ids=None):
    if repr_1.shape != repr_2.shape or repr_1.ndim != 2 or len(repr_1) == 0:
        raise ValueError("Expected equal nonempty [B,D] representations")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    # FP32 avoids half-precision overflow/underflow at small temperatures.
    with torch.autocast(device_type=repr_1.device.type, enabled=False):
        z1, z2 = F.normalize(repr_1.float(), dim=-1), F.normalize(repr_2.float(), dim=-1)
        similarity = z1 @ z2.T / temperature
        labels = torch.arange(len(z1), device=z1.device)
        if session_ids is not None:
            if session_ids.shape != labels.shape:
                raise ValueError("Session IDs must match the batch")
            same_session = session_ids[:, None].eq(session_ids[None])
            same_session.fill_diagonal_(False)
            similarity = similarity.masked_fill(same_session, float("-inf"))
        return .5 * (F.cross_entropy(similarity, labels) + F.cross_entropy(similarity.T, labels))


def train_contrastive_sasrec(seed, output_dir, method="head", config=None,
                            processed_dir=PROCESSED_DIGI_DIR, device_str=None,
                            train_limit=None, validation_limit=None, evaluate_test=False):
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    cfg = copy.deepcopy(config) if config is not None else load_experiment_config("sasrec")
    if cfg.get("model_name") != "sasrec":
        raise ValueError("Contrastive study requires the SASRec architecture")
    training = cfg["training"]
    settings = cfg.get("contrastive", {})
    epochs = int(training.get("epochs", 15))
    cl_weight = 0. if method == "baseline" else float(settings.get("weight", .15))
    temperature = float(settings.get("temperature", .1))
    head_probability = float(settings.get("mask_prob_head", .4))
    other_probability = float(settings.get("mask_prob_other", .1))
    uniform_setting = settings.get("mask_prob_uniform")
    uniform_probability = None if uniform_setting is None else float(uniform_setting)
    if epochs < 1 or cl_weight < 0 or temperature <= 0:
        raise ValueError("Invalid training/contrastive parameters")
    if not all(0 <= p <= 1 for p in (head_probability, other_probability)) or (uniform_probability is not None and not 0 <= uniform_probability <= 1):
        raise ValueError("Mask probabilities must be in [0,1]")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    logger = create_run_logger(output / "training.log", f"contrastive.{method}.{seed}")
    set_global_seed(seed)
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    mapping = load_item_mapping(processed_dir)
    train = pd.read_parquet(processed_dir / "train_examples.parquet")
    validation = pd.read_parquet(processed_dir / "validation_examples.parquet")
    for limit in (train_limit, validation_limit):
        if limit is not None and limit < 1:
            raise ValueError("Sample limits must be positive")
    if train_limit is not None and train_limit < len(train):
        train = train.sample(train_limit, random_state=seed).sort_index()
    if validation_limit is not None and validation_limit < len(validation):
        validation = validation.sample(validation_limit, random_state=seed + 1).sort_index()
    if train.empty or validation.empty:
        raise ValueError("Empty train/validation split")
    model = build_model_from_config(cfg, mapping, device)
    model._experiment_max_seq_length = int(cfg.get("max_seq_length", 50))
    for frame in (train, validation):
        targets = torch.tensor(frame.target_internal_item_id.to_numpy(copy=True), device=device)
        if not model.trainable_warm_mask[targets].all():
            raise ValueError("Non-warm target in the main protocol")
    frozen = ~model.trainable_warm_mask
    initial_frozen = model.item_embedding.weight.detach()[frozen].clone()
    head_mask = torch.zeros(model.num_items_with_pad, dtype=torch.bool, device=device)
    head_mask[torch.tensor(mapping.loc[mapping.popularity_group.eq("popular"), "internal_item_id"].to_numpy(copy=True), device=device)] = True
    dataset = SessionPrefixDataset(train, model._experiment_max_seq_length)
    if uniform_probability is None:
        # Match the expected number of deleted tokens, using TRAIN positions
        # only and excluding the protected final event.
        eligible_positions = torch.arange(dataset.sequences.shape[1])[None] < (dataset.lengths - 1)[:, None]
        item_counts = torch.bincount(dataset.sequences[eligible_positions], minlength=model.num_items_with_pad)
        denominator = int(item_counts.sum())
        head_share = float(item_counts[head_mask.cpu()].sum()) / denominator if denominator else 0.
        uniform_probability = head_share * head_probability + (1 - head_share) * other_probability
        del eligible_positions, item_counts
    cfg.setdefault("contrastive", {})["resolved_uniform_mask_probability"] = uniform_probability
    validation_dataset = SessionPrefixDataset(validation, model._experiment_max_seq_length)
    generator = torch.Generator().manual_seed(seed)
    augmentation_generator = torch.Generator(device=device).manual_seed(seed + 100_003)
    loader = DataLoader(dataset, batch_size=int(training.get("train_batch_size", 512)),
        shuffle=True, generator=generator, num_workers=0, pin_memory=device.type == "cuda")
    # Honor the SASRec configuration (AMP is disabled there for stability),
    # rather than enabling FP16 merely because a GPU is present.
    amp = bool(training.get("use_amp", False)) and device.type == "cuda"
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training.get("learning_rate", .001)),
                                 weight_decay=float(training.get("weight_decay", 0)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    cfg.update(seed=int(seed), contrastive_method=method)
    cfg["experiment"] = {"device": str(device), "amp": amp, "train_examples": len(train),
        "validation_examples": len(validation), "train_limit": train_limit, "validation_limit": validation_limit,
        "evaluate_test": evaluate_test, "cold_protocol": "frozen_and_masked", "mask_semantics": "compact_item_deletion_keep_last",
        "status": "smoke_test" if train_limit or validation_limit else "full_training"}
    save_yaml(cfg, output / "config.yaml")
    mapping.to_csv(output / "item_embedding_mapping.csv", index=False)
    pd.DataFrame({"source_row": train.index, "session_id": train.session_id.to_numpy()}).to_csv(output / "train_sample.csv.gz", index=False)
    monitor = cfg.get("early_stopping", {}).get("monitor", "mrr@20")
    patience = int(cfg.get("early_stopping", {}).get("patience", 4))
    min_delta = float(cfg.get("early_stopping", {}).get("min_delta", .0001))
    topk = int(cfg["evaluation"].get("topk", 20))
    if not 1 <= topk <= int(model.trainable_warm_mask.sum()):
        raise ValueError("topk exceeds warm catalog size")
    if monitor not in {f"{name}@{topk}" for name in ("recall", "mrr", "ndcg")}:
        raise ValueError("Early-stopping monitor must use the configured topk")
    if patience < 1 or min_delta < 0:
        raise ValueError("Invalid early-stopping parameters")
    evaluation_batch_size = int(training.get("eval_batch_size", 512))
    best_score, best_epoch, stale = -float("inf"), 0, 0
    history = []
    started = time.perf_counter()
    logger.info("START method=%s seed=%s device=%s amp=%s train=%s validation=%s", method, seed, device, amp, len(train), len(validation))
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_ce, total_cl, samples = 0., 0., 0
        for batch in loader:
            sequence = batch["item_seq"].to(device)
            lengths = batch["item_seq_len"].to(device)
            targets = batch["target"].to(device)
            session_ids = torch.as_tensor(dataset.session_ids[batch["query_id"].numpy()], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                session = model.encode_session(sequence, lengths)
                # Evaluate CE in FP32; -1e9 cannot be represented in FP16.
                logits = (session @ model.item_embedding.weight.T).float()
                logits = logits.masked_fill(~model.trainable_warm_mask[None], -1e9)
                ce = F.cross_entropy(logits, targets)
                contrastive = session.float().sum() * 0
                if cl_weight > 0:
                    p_head, p_other = ((uniform_probability, uniform_probability) if method == "uniform"
                                      else (head_probability, other_probability))
                    v1, l1, v2, l2 = generate_popularity_aware_views(sequence, lengths, head_mask,
                        p_head, p_other, augmentation_generator)
                    contrastive = compute_infonce_contrastive_loss(model.encode_session(v1, l1), model.encode_session(v2, l2),
                        temperature, session_ids if settings.get("exclude_same_session_negatives", True) else None)
                loss = ce + cl_weight * contrastive
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch={epoch}, method={method}; see {output}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip = float(training.get("grad_clip_norm", 5))
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                model.item_embedding.weight[frozen] = initial_frozen
            count = len(targets)
            total_ce += float(ce.detach()) * count
            total_cl += float(contrastive.detach()) * count
            samples += count
        predicted, _ = predict_topk_for_dataset(model, validation_dataset, device, evaluation_batch_size, topk)
        metrics = compute_ranking_metrics_from_predictions(predicted, validation_dataset.targets, k=topk)
        score = float(metrics[monitor])
        improved = score > best_score + min_delta
        if improved:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "config": cfg, "epoch": epoch,
                        "best_validation_score": score, "monitor": monitor}, output / "checkpoint.pth")
        else:
            stale += 1
        row = {"epoch": epoch, "ce_loss": total_ce / samples, "contrastive_loss": total_cl / samples,
               "validation_score": score, "is_best": improved, "seconds": time.perf_counter() - epoch_start,
               **{f"validation_{key}": value for key, value in metrics.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        logger.info("epoch=%s CE=%.5f CL=%.5f val_%s=%.5f seconds=%.1f%s", epoch, row["ce_loss"], row["contrastive_loss"],
                    monitor, score, row["seconds"], " BEST" if improved else "")
        if stale >= patience:
            logger.info("Early stopping; best epoch=%s", best_epoch)
            break
    model.load_state_dict(torch.load(output / "checkpoint.pth", map_location=device, weights_only=False)["model_state_dict"])
    drift = float((model.item_embedding.weight.detach()[frozen] - initial_frozen).abs().max())
    if drift != 0:
        raise AssertionError(f"Frozen embedding drift: {drift}")
    result = {"model": "sasrec", "method": method, "seed": seed, "best_epoch": best_epoch,
        "monitor": monitor, "best_validation_score": best_score, "frozen_embedding_max_abs_drift": drift,
        "experiment": cfg["experiment"], "output_dir": str(output.resolve()), "seconds": time.perf_counter() - started}
    result["validation"] = _evaluate(model, validation, mapping, device, evaluation_batch_size, topk, output, "validation")
    if evaluate_test:
        test = pd.read_parquet(processed_dir / "test_examples.parquet")
        if not model.trainable_warm_mask[torch.tensor(test.target_internal_item_id.to_numpy(copy=True), device=device)].all():
            raise ValueError("Non-warm test target")
        result["test"] = _evaluate(model, test, mapping, device, evaluation_batch_size, topk, output, "test")
    result["seconds"] = time.perf_counter() - started
    save_json(result, output / "metrics.json")
    logger.info("COMPLETE %s", output)
    return result
