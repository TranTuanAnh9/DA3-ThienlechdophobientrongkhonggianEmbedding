"""Report measured role-gradient effects against CE, simple scaling and epoch 0.

No test labels are read. Compare identical validation queries, cluster-resample
sessions, and explicitly record small effects and absent gains.
"""
from pathlib import Path
import argparse
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from btl_k03.models import build_model_from_config
from btl_k03.artifacts import load_item_embeddings
from btl_k03.topology.topo_trainer import _evaluate, _sample
from btl_k03.utils import load_item_mapping, load_yaml
from btl_k03.statistics import clustered_delta


def report(roots, output, iterations=1000):
    output.mkdir(parents=True, exist_ok=False)
    comparisons = []
    files = {}
    for root in roots:
        table = pd.read_csv(root / "comparison.csv")
        comparisons.append(table)
        for model, method, seed in table[["model", "method", "seed"]].drop_duplicates().itertuples(index=False, name=None):
            key = (model, method, seed)
            if key in files:
                raise ValueError(f"Duplicate experiment: {key}")
            files[key] = root / model / method / f"seed_{seed}"
    combined = pd.concat(comparisons, ignore_index=True)
    combined.to_csv(output / "all_measured_results.csv", index=False)
    mapping = load_item_mapping()
    processed = ROOT / "data/processed/diginetica"
    validation = pd.read_parquet(processed / "validation_examples.parquet")
    models = combined[["model", "seed"]].drop_duplicates()
    differences, rows, diagnostics = [], [], []
    rng = np.random.default_rng(20261121)
    for model, seed in models.itertuples(index=False, name=None):
        baseline = files[(model, "baseline", seed)]
        config = load_yaml(baseline / "config.yaml")
        reference = None
        if config["experiment"].get("initial_checkpoint"):
            checkpoint_path = Path(config["experiment"]["initial_checkpoint"])
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            original = build_model_from_config(checkpoint["config"], mapping, torch.device("cpu"))
            original.load_state_dict(checkpoint["model_state_dict"])
            original._experiment_max_seq_length = int(config.get("max_seq_length", 50))
            reference = output / "untouched_checkpoint" / model / f"seed_{seed}"
            reference.mkdir(parents=True)
            sample = _sample(validation, config["experiment"]["validation_limit"], seed + 1)
            initial_metrics = _evaluate(original, sample, mapping, torch.device("cpu"), 128, 20, reference, "validation")
            (reference / "metrics.json").write_text(json.dumps(initial_metrics, indent=2), encoding="utf-8")
        available = {method: path for (name, method, item_seed), path in files.items() if name == model and item_seed == seed}
        baseline_embeddings = load_item_embeddings(baseline)
        queries = {method: pd.read_csv(path / "validation_per_query.csv.gz") for method, path in available.items()}
        if reference is not None:
            queries["untouched"] = pd.read_csv(reference / "validation_per_query.csv.gz")
        controls = [method for method in ("baseline", "role_input_scale", "role_tangent", "untouched") if method in queries]
        for method, query in queries.items():
            for group in ("all_warm", "rare"):
                subset = query if group == "all_warm" else query[query.group.eq(group)]
                base = queries["baseline"] if group == "all_warm" else queries["baseline"][queries["baseline"].group.eq(group)]
                # Explicit merge guards against positional comparisons of
                # different samples and retains paired hit transitions.
                aligned = base.merge(subset, on=["source_row", "session_id", "target", "group"],
                    suffixes=("_base", "_new"), validate="one_to_one")
                if len(aligned) != len(base) or len(aligned) != len(subset):
                    raise ValueError("Validation query samples differ")
                rows.append({"model": model, "seed": seed, "method": method, "group": group,
                    "n_queries": len(subset), "hits": int(subset.hit.sum()), "recall": float(subset.hit.mean()),
                    "mrr": float(subset.reciprocal_rank.mean()), "new_hits_vs_ce": int(((aligned.hit_base == 0) & (aligned.hit_new == 1)).sum()),
                    "lost_hits_vs_ce": int(((aligned.hit_base == 1) & (aligned.hit_new == 0)).sum())})
            if method in available:
                history = pd.read_csv(available[method] / "training_history.csv")
                diagnostics.append({"model": model, "seed": seed, "method": method,
                    "adjusted_item_batch_events": float(history.get("adjusted_items", pd.Series([0])).sum()),
                    "input_active_item_batch_events": float(history.get("input_active_items", pd.Series([0])).sum()),
                    "ph_h0_mean": float(history.ph_h0.mean()), "ph_h1_mean": float(history.ph_h1.mean()),
                    "rare_mean_embedding_distance_from_ce": float(np.linalg.norm(
                        (load_item_embeddings(available[method]) - baseline_embeddings)[mapping.popularity_group.eq("rare").to_numpy()], axis=1).mean())})
            if method not in ("role_ph", "ph_h0_h1", "role_input_scale"):
                continue
            for control in controls:
                if method == control:
                    continue
                for group, metric in (("rare", "hit"), ("all_warm", "hit"), ("rare", "reciprocal_rank")):
                    delta = clustered_delta(queries[control], query, metric, group, iterations, rng)
                    differences.append({"model": model, "seed": seed, "method": method, "control": control,
                                        "group": group, "metric": metric, **delta})
        print(f"Evaluated paired controls: {model} seed={seed}", flush=True)
    pd.DataFrame(rows).to_csv(output / "ranking_effects.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "mechanism_diagnostics.csv", index=False)
    delta_frame = pd.DataFrame(differences)
    delta_frame.to_csv(output / "paired_session_bootstrap.csv", index=False)
    decisive = delta_frame[delta_frame.method.eq("role_ph") & delta_frame.control.isin(["baseline", "role_input_scale", "untouched"])
                            & delta_frame.group.eq("rare") & delta_frame.metric.eq("hit")]
    passes = bool(len(decisive)) and decisive.ci_low.gt(0).all()
    verdict = {"method": "role_ph", "status": "exploratory_positive_signal" if passes else "added_value_not_demonstrated",
        "criterion": "positive lower session-bootstrap rare Recall bound against CE, input scaling and untouched checkpoint for every measured model/seed",
        "models": models.to_dict(orient="records"), "test_used": False,
        "scope": "Small-sample fine-tuning from validation-selected checkpoints; not a full-training or confirmatory result",
        "inference": "Architecture and inference parameter count unchanged; inference latency not measured",
        "novelty": "Not established; code implementation is not proof of a new scientific contribution"}
    (output / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    plot = delta_frame[delta_frame.control.eq("baseline") & delta_frame.group.eq("rare") & delta_frame.metric.eq("hit")]
    if len(plot):
        figure, axis = plt.subplots(figsize=(12, max(4, len(plot) * .25)))
        for index, row in enumerate(plot.itertuples(index=False)):
            axis.errorbar(row.delta * 100, index, xerr=[[max(0, row.delta - row.ci_low) * 100], [max(0, row.ci_high - row.delta) * 100]],
                fmt="o", capsize=3, color="tab:blue" if row.method == "role_ph" else "tab:gray")
        axis.set_yticks(range(len(plot)), [f"{r.model} / {r.seed} / {r.method}" for r in plot.itertuples(index=False)])
        axis.axvline(0, color="black", linestyle="--", linewidth=1)
        axis.set(xlabel="Rare Recall difference vs CE (percentage points), paired session bootstrap 95% CI",
                 title="Measured exploratory effects; intervals do not include training uncertainty")
        figure.tight_layout()
        figure.savefig(output / "rare_recall_effects.png", dpi=180)
        plt.close(figure)
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    torch.set_num_threads(2)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        report(args.roots, args.output, args.iterations)
