"""Audit matched embeddings without prewritten improvement claims.

Two independent t-SNE fits are illustrations, not a measure of isotropy or
displacement. Quantitative statistics use the original warm embedding space.
Default candidate is the LEGACY cosine run, not the new actual-PH method.
"""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import sys
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from btl_k03.artifacts import load_item_embeddings


def load(directory):
    embedding = load_item_embeddings(directory)
    mapping_path = directory / "item_embedding_mapping.csv"
    if not mapping_path.exists():
        # Legacy runs did not save their raw-ID mapping. Use the project's
        # catalog explicitly, verify the archived internal mask/weights, and
        # record that raw-ID mapping provenance is inferred rather than saved.
        mapping_path = ROOT / "data/processed/diginetica/item_mapping.csv"
        state = torch.load(directory / "checkpoint.pth", map_location="cpu", weights_only=False)["model_state_dict"]
        np.testing.assert_allclose(state["item_embedding.weight"].numpy()[1:], embedding)
        expected = pd.read_csv(mapping_path).sort_values("internal_item_id").is_trainable_warm.to_numpy()
        np.testing.assert_array_equal(state["trainable_warm_mask"].numpy()[1:], expected)
    mapping = pd.read_csv(mapping_path).sort_values("internal_item_id").reset_index(drop=True)
    mapping.attrs["mapping_source"] = str(mapping_path)
    if len(embedding) != len(mapping) or not np.array_equal(mapping.internal_item_id, np.arange(1, len(mapping) + 1)):
        raise ValueError("Embedding rows must correspond to internal IDs 1..N, excluding padding")
    if "embedding_row_index" in mapping and not np.array_equal(mapping.embedding_row_index, np.arange(len(mapping))):
        raise ValueError("Saved row mapping is inconsistent")
    return embedding, mapping


def metrics(matrix):
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular ** 2
    proportions = energy / energy.sum()
    positive = proportions[proportions > 0]
    return {"centered_pc1_variance_share": float(proportions[0]),
            "centered_pc5_variance_share": float(proportions[:5].sum()),
            "centered_effective_rank": float(np.exp(-(positive * np.log(positive)).sum())),
            "mean_unit_vector_norm": float(np.linalg.norm(normalize(matrix).mean(axis=0)))}, proportions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "runs/sasrec/seed_2024")
    parser.add_argument("--candidate", type=Path, default=ROOT / "runs/sasrec_topology/seed_2024")
    parser.add_argument("--candidate-label", default="Legacy cosine regularizer (not actual PH)")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/figures/tsne_embedding_manifold_audited.png")
    parser.add_argument("--per-group", type=int, default=700)
    args = parser.parse_args()
    baseline, mapping = load(args.baseline)
    candidate, other = load(args.candidate)
    if not mapping[["internal_item_id", "item_id", "popularity_group", "is_trainable_warm"]].equals(
        other[["internal_item_id", "item_id", "popularity_group", "is_trainable_warm"]]
    ):
        raise ValueError("Baseline/candidate catalogs or group definitions differ")
    warm = mapping.is_trainable_warm.to_numpy(dtype=bool)
    sample = pd.concat([part.sample(min(args.per_group, len(part)), random_state=42)
                        for group, part in mapping.loc[warm].groupby("popularity_group")
                        if group in ("popular", "medium", "rare")])
    rows = sample.internal_item_id.to_numpy() - 1
    if len(rows) <= 35:
        raise ValueError("Need at least 36 sampled warm items for perplexity 35")
    groups = sample.popularity_group.to_numpy()
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    results, spectra = [], {}
    labels = ("Baseline SASRec", args.candidate_label)
    for column, (label, values) in enumerate(zip(labels, (baseline, candidate))):
        unit = normalize(values[warm])
        for space, points in (("raw", values[warm]), ("unit", unit)):
            measured, spectrum = metrics(points)
            results.append({"model": label, "space": space, "n_warm_items": int(warm.sum()), **measured})
            if space == "unit":
                spectra[label] = spectrum
        points = TSNE(n_components=2, perplexity=35, random_state=42, max_iter=1000,
                      init="pca", learning_rate="auto").fit_transform(normalize(values[rows]))
        for group, color in (("popular", "#d64b40"), ("medium", "#e9a326"), ("rare", "#26945c")):
            selected = groups == group
            axes[0, column].scatter(points[selected, 0], points[selected, 1], color=color, s=8, alpha=.55, label=group)
        axes[0, column].set(title=label, xlabel="t-SNE 1 (independent fit)", ylabel="t-SNE 2")
        axes[0, column].legend()
    for label, spectrum in spectra.items():
        axes[1, 0].plot(np.arange(1, len(spectrum) + 1), spectrum, label=label)
        axes[1, 1].plot(np.arange(1, len(spectrum) + 1), np.cumsum(spectrum), label=label)
    axes[1, 0].set(title="Measured angular spectrum in original 64-D space", xlabel="Principal component", ylabel="Centered variance share")
    axes[1, 1].set(title="Measured cumulative angular variance", xlabel="Number of components", ylabel="Cumulative share")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].legend(fontsize=8)
    figure.suptitle("Matched warm items; L2-normalized t-SNE is illustrative\nIndependent 2-D fits do not establish isotropy, semantic quality or ranking improvement", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, .94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    pd.DataFrame(results).to_csv(args.output.with_suffix(".metrics.csv"), index=False)
    (args.output.with_suffix(".provenance.json")).write_text(json.dumps({"baseline": str(args.baseline),
        "candidate": str(args.candidate), "candidate_label": args.candidate_label,
        "baseline_mapping_source": mapping.attrs["mapping_source"],
        "candidate_mapping_source": other.attrs["mapping_source"],
        "legacy_mapping_note": "Global catalog inferred if legacy run has no saved mapping; checkpoint mask/weights verified",
        "sampled_internal_ids": sample.internal_item_id.tolist(), "sample_rows": rows.tolist(),
        "includes_cold": False, "tsne_fits": "independent; no cross-panel coordinate comparison",
        "quantitative_space": "all warm embeddings before dimension reduction"}, indent=2), encoding="utf-8")
    print(pd.DataFrame(results).to_string(index=False))
    print(args.output.resolve())


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
