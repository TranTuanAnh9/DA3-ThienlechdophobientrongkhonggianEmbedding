import numpy as np
import pandas as pd

from btl_k03.embedding_analysis import (
    attach_item_accuracy,
    build_query_and_item_accuracy_tables,
    fit_geometry_accuracy_models,
    fit_geometry_exposure_models,
    summarize_geometry_and_outcomes_by_group,
)


def _synthetic_item_table(n_items: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(2024)
    item_ids = np.arange(1, n_items + 1)
    groups = np.repeat(["rare", "medium", "popular"], n_items // 3)
    train_freq = np.concatenate([
        rng.integers(5, 7, n_items // 3),
        rng.integers(7, 27, n_items // 3),
        rng.integers(27, 80, n_items // 3),
    ])
    return pd.DataFrame({
        "model": "toy",
        "seed": 2024,
        "internal_item_id": item_ids,
        "is_trainable_warm": True,
        "popularity_group": groups,
        "train_freq": train_freq,
        "embedding_norm": 1.5 - 0.01 * train_freq + rng.normal(0, 0.02, n_items),
        "distance_to_category_centroid": 0.9 - 0.004 * train_freq + rng.normal(0, 0.01, n_items),
        "cosine_to_category_centroid": 0.4 + 0.004 * train_freq + rng.normal(0, 0.01, n_items),
        "rec_count": np.maximum(0, train_freq * 2 + rng.integers(-3, 4, n_items)),
    })


def test_build_query_and_item_accuracy_tables() -> None:
    examples = pd.DataFrame({
        "session_id": [10, 11, 12],
        "target_internal_item_id": [1, 2, 1],
        "target_popularity_group": ["rare", "medium", "rare"],
    })
    top20 = pd.DataFrame({
        "session_id": [10, 10, 11, 11, 12, 12],
        "query_id": [0, 0, 1, 1, 2, 2],
        "rank": [1, 2, 1, 2, 1, 2],
        "internal_item_id": [1, 3, 3, 4, 2, 1],
    })

    queries, items = build_query_and_item_accuracy_tables(examples, top20, topk=2)

    assert queries["hit@2"].tolist() == [True, False, True]
    item_1 = items.loc[items["internal_item_id"] == 1].iloc[0]
    item_2 = items.loc[items["internal_item_id"] == 2].iloc[0]
    assert item_1["target_count"] == 2
    assert item_1["hit_count"] == 2
    assert item_1["item_recall@2"] == 1.0
    assert item_2["target_count"] == 1
    assert item_2["hit_count"] == 0


def test_group_summary_reproduces_micro_recall_and_exposure() -> None:
    item_table = _synthetic_item_table()
    accuracy = pd.DataFrame({
        "internal_item_id": np.arange(1, 31),
        "target_count": 2,
        "hit_count": np.tile([0, 1, 2], 10),
        "item_recall@20": np.tile([0.0, 0.5, 1.0], 10),
    })
    enriched = attach_item_accuracy(item_table, accuracy, topk=20)
    summary = summarize_geometry_and_outcomes_by_group(enriched, topk=20)

    assert set(summary["popularity_group"]) == {"popular", "medium", "rare"}
    assert np.isclose(summary["exposure_share_pct"].sum(), 100.0)
    assert summary["target_count"].sum() == 60
    assert np.all((summary["recall@20"] >= 0) & (summary["recall@20"] <= 1))


def test_controlled_models_return_expected_rows() -> None:
    item_table = _synthetic_item_table()
    exposure_models = fit_geometry_exposure_models(item_table)
    assert len(exposure_models) == 4
    assert set(exposure_models["geometry_variable"]) == {
        "none",
        "embedding_norm",
        "distance_to_category_centroid",
        "cosine_to_category_centroid",
    }

    query_rows = []
    for query_id in range(120):
        target_id = query_id % 30 + 1
        query_rows.append({
            "query_id": query_id,
            "session_id": query_id,
            "target_internal_item_id": target_id,
            "hit@20": bool(query_id % 3),
        })
    accuracy_models = fit_geometry_accuracy_models(
        pd.DataFrame(query_rows), item_table, topk=20
    )
    assert len(accuracy_models) == 4
    assert accuracy_models["n_queries"].eq(120).all()
    assert np.isfinite(accuracy_models["log_loss"]).all()
