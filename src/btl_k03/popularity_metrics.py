from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_gini_coefficient(counts: np.ndarray | list[float] | pd.Series) -> float:
    """
    Tính hệ số Gini chuẩn trên vector tần suất được gợi ý (rec_count) của toàn bộ catalog hợp lệ,
    bao gồm cả các item có rec_count = 0 (theo đúng yêu cầu Mục 9.3 tài liệu đề tài).
    Giá trị thuộc [0, 1 - 1/N]: gần 0 là phân bổ đồng đều, gần 1 là tập trung cực đoan.
    """
    values = np.asarray(counts, dtype=np.float64).ravel()
    if values.size == 0:
        return 0.0
    if np.any(values < 0):
        raise ValueError("Tần suất gợi ý (rec_count) không được âm.")
    total = float(values.sum())
    if total == 0.0 or values.size == 1:
        return 0.0

    sorted_vals = np.sort(values)
    n = sorted_vals.size
    index = np.arange(1, n + 1, dtype=np.float64)
    gini = float((np.sum((2.0 * index - n - 1.0) * sorted_vals)) / (n * total))
    return max(0.0, gini)


def compute_lorenz_curve(
    counts: np.ndarray | list[float] | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Trả về tọa độ (item_share_cum, exposure_share_cum) để vẽ đường cong Lorenz."""
    values = np.sort(np.asarray(counts, dtype=np.float64).ravel())
    n = values.size
    if n == 0 or values.sum() == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    x = np.concatenate([[0.0], np.arange(1, n + 1, dtype=np.float64) / n])
    y = np.concatenate([[0.0], np.cumsum(values) / values.sum()])
    return x, y


def compute_catalog_coverage(
    recommended_items: np.ndarray | list[int] | pd.Series,
    catalog_size: int,
) -> float:
    """
    Tính Catalog Coverage@K = (số item khác nhau xuất hiện ít nhất 1 lần trong các danh sách top-K)
    chia cho tổng số item hợp lệ trong catalog.
    """
    if catalog_size <= 0:
        raise ValueError("catalog_size phải lớn hơn 0.")
    unique_items = pd.unique(np.asarray(recommended_items).ravel())
    return float(len(unique_items) / float(catalog_size))


def compute_arp(
    recommended_items: np.ndarray | list[int] | pd.Series,
    item_popularity_map: dict[int, float] | pd.Series,
    use_log1p: bool = False,
) -> float:
    """
    Tính Average Recommendation Popularity (ARP):
    Độ phổ biến trung bình trong tập train của tất cả các vị trí item xuất hiện trong top-K.
    """
    rec_array = np.asarray(recommended_items).ravel()
    if rec_array.size == 0:
        return 0.0

    if isinstance(item_popularity_map, pd.Series):
        pop_values = item_popularity_map.reindex(rec_array).fillna(0.0).to_numpy(dtype=np.float64)
    else:
        pop_values = np.array(
            [float(item_popularity_map.get(int(item), 0.0)) for item in rec_array],
            dtype=np.float64,
        )

    if use_log1p:
        pop_values = np.log1p(pop_values)
    return float(np.mean(pop_values))


def compute_ranking_metrics_from_predictions(
    top_k_ids: np.ndarray,
    target_ids: np.ndarray,
    k: int = 20,
) -> dict[str, float]:
    """
    Tính Recall@K (Hit Rate@K), MRR@K và NDCG@K từ ma trận dự đoán top_k_ids (N, K)
    và vector nhãn thực tế target_ids (N,).
    """
    preds = np.asarray(top_k_ids)[:, :k]
    targets = np.asarray(target_ids).reshape(-1, 1)

    if preds.shape[0] == 0:
        return {f"recall@{k}": 0.0, f"mrr@{k}": 0.0, f"ndcg@{k}": 0.0}

    hits_matrix = preds == targets
    hit_any = hits_matrix.any(axis=1)
    recall = float(np.mean(hit_any))

    ranks = np.argmax(hits_matrix, axis=1) + 1  # 1-indexed rank
    mrr_scores = np.where(hit_any, 1.0 / ranks, 0.0)
    ndcg_scores = np.where(hit_any, 1.0 / np.log2(ranks + 1.0), 0.0)

    return {
        f"recall@{k}": recall,
        f"mrr@{k}": float(np.mean(mrr_scores)),
        f"ndcg@{k}": float(np.mean(ndcg_scores)),
    }


def compute_stratified_ranking_metrics(
    top_k_ids: np.ndarray,
    target_ids: np.ndarray,
    target_groups: np.ndarray | pd.Series,
    k: int = 20,
) -> pd.DataFrame:
    """Tính Recall@K, MRR@K, NDCG@K tổng thể và tách theo từng nhóm target_popularity_group."""
    preds = np.asarray(top_k_ids)[:, :k]
    targets = np.asarray(target_ids)
    groups = np.asarray(target_groups)

    rows = []
    overall = compute_ranking_metrics_from_predictions(preds, targets, k=k)
    rows.append(
        {
            "target_group": "all_warm",
            "num_examples": int(len(targets)),
            "example_share_pct": 100.0,
            **overall,
        }
    )

    for grp in ("popular", "medium", "rare"):
        mask = groups == grp
        count = int(mask.sum())
        if count > 0:
            grp_metrics = compute_ranking_metrics_from_predictions(
                preds[mask], targets[mask], k=k
            )
        else:
            grp_metrics = {f"recall@{k}": 0.0, f"mrr@{k}": 0.0, f"ndcg@{k}": 0.0}
        rows.append(
            {
                "target_group": grp,
                "num_examples": count,
                "example_share_pct": float(count * 100.0 / max(1, len(targets))),
                **grp_metrics,
            }
        )

    return pd.DataFrame(rows)


def evaluate_top20_popularity_bias(
    top20_df: pd.DataFrame,
    item_mapping: pd.DataFrame,
    k: int = 20,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Tính toàn bộ chỉ số thiên lệch độ phổ biến ở đầu ra từ bảng top20.parquet và item_mapping.csv:
    - Coverage@20 (trên warm candidate catalog 29,936 item và trên full model catalog 37,583 item)
    - Gini@20 (bao gồm cả các item có rec_count = 0)
    - ARP@20 (theo tần suất train gốc và theo log1p(train_frequency))
    - Bảng item_rec_stats (rec_count và exposure_share cho từng item)
    - Bảng group_exposure_df (tỷ trọng lượt hiển thị theo nhóm popular/medium/rare/cold)
    """
    filtered_top = top20_df.loc[top20_df["rank"] <= k].copy()
    total_Recommended_slots = len(filtered_top)
    num_queries = filtered_top["query_id"].nunique() if "query_id" in filtered_top.columns else int(total_Recommended_slots // max(1, k))

    rec_counts_series = filtered_top["item_id"].value_counts()

    catalog = item_mapping[
        [
            "internal_item_id",
            "item_id",
            "category_id",
            "internal_category_id",
            "train_frequency",
            "popularity_group",
            "is_controlled_cold",
            "is_trainable_warm",
            "candidate_for_main_evaluation",
        ]
    ].copy()

    catalog["rec_count"] = (
        catalog["item_id"].map(rec_counts_series).fillna(0).astype(np.int64)
    )
    catalog["exposure_share"] = (
        catalog["rec_count"] / float(max(1, total_Recommended_slots))
    )

    warm_catalog = catalog.loc[catalog["candidate_for_main_evaluation"]].copy()
    n_warm_items = len(warm_catalog)
    n_full_items = len(catalog)

    unique_rec_warm = int((warm_catalog["rec_count"] > 0).sum())
    unique_rec_full = int((catalog["rec_count"] > 0).sum())

    coverage_warm = float(unique_rec_warm / max(1, n_warm_items))
    coverage_full = float(unique_rec_full / max(1, n_full_items))

    gini_warm = compute_gini_coefficient(warm_catalog["rec_count"].to_numpy())
    gini_full = compute_gini_coefficient(catalog["rec_count"].to_numpy())

    freq_lookup = catalog.set_index("item_id")["train_frequency"]
    arp_raw = compute_arp(filtered_top["item_id"], freq_lookup, use_log1p=False)
    arp_log1p = compute_arp(filtered_top["item_id"], freq_lookup, use_log1p=True)

    train_total_interactions = float(warm_catalog["train_frequency"].sum())
    group_rows = []
    for grp in ("popular", "medium", "rare", "cold"):
        sub = catalog.loc[catalog["popularity_group"] == grp]
        item_cnt = int(len(sub))
        rec_sum = int(sub["rec_count"].sum())
        train_sum = int(sub["train_frequency"].sum())
        active_items = int((sub["rec_count"] > 0).sum())
        exposure_pct = float(rec_sum * 100.0 / max(1, total_Recommended_slots))
        train_share_pct = float(train_sum * 100.0 / max(1.0, train_total_interactions))
        catalog_share_warm_pct = (
            float(item_cnt * 100.0 / max(1, n_warm_items)) if grp != "cold" else 0.0
        )
        group_rows.append(
            {
                "popularity_group": grp,
                "catalog_items": item_cnt,
                "warm_catalog_share_pct": catalog_share_warm_pct,
                "train_interactions": train_sum,
                "train_interaction_share_pct": train_share_pct,
                "recommended_slots": rec_sum,
                "exposure_share_pct": exposure_pct,
                "exposure_to_train_ratio": (
                    float(exposure_pct / train_share_pct) if train_share_pct > 0 else 0.0
                ),
                "unique_items_recommended": active_items,
                "within_group_coverage_pct": float(
                    active_items * 100.0 / max(1, item_cnt)
                ),
                "mean_rec_count_per_item": float(sub["rec_count"].mean()) if item_cnt > 0 else 0.0,
                "median_rec_count_per_item": float(sub["rec_count"].median()) if item_cnt > 0 else 0.0,
            }
        )

    group_exposure_df = pd.DataFrame(group_rows)

    summary = {
        "topk": int(k),
        "num_queries": int(num_queries),
        "total_recommendation_slots": int(total_Recommended_slots),
        "warm_catalog_size": int(n_warm_items),
        "full_catalog_size": int(n_full_items),
        "unique_recommended_items_warm": unique_rec_warm,
        "unique_recommended_items_full": unique_rec_full,
        f"coverage_warm@{k}": coverage_warm,
        f"coverage_full@{k}": coverage_full,
        f"gini_warm@{k}": gini_warm,
        f"gini_full@{k}": gini_full,
        f"arp@{k}": arp_raw,
        f"arp_log1p@{k}": arp_log1p,
        "popular_group_exposure_share_pct": float(
            group_exposure_df.loc[
                group_exposure_df["popularity_group"] == "popular", "exposure_share_pct"
            ].iloc[0]
        ),
        "rare_group_exposure_share_pct": float(
            group_exposure_df.loc[
                group_exposure_df["popularity_group"] == "rare", "exposure_share_pct"
            ].iloc[0]
        ),
    }

    return summary, catalog, group_exposure_df


def compute_cold_start_test_diagnostic(
    warm_metrics: dict[str, float],
    test_events_full: pd.DataFrame,
    warm_test_examples_count: int,
    k: int = 20,
) -> dict[str, Any]:
    """
    Phân tích đối chứng giữa đánh giá trên tập Warm-item chính thức và tập Full Test (có chứa
    20% Controlled Cold + Natural Cold items). Giúp trả lời chặt chẽ câu hỏi của giảng viên về
    hướng đánh giá Warm-item vs. Cold-start.
    """
    ordered = test_events_full.sort_values(
        ["session_id", "timeframe", "row_order"], kind="mergesort"
    )
    session_lengths = ordered.groupby("session_id").size()
    valid_sessions = set(session_lengths[session_lengths >= 2].index)
    ordered = ordered.loc[ordered["session_id"].isin(valid_sessions)].copy()

    ordered["pos"] = ordered.groupby("session_id").cumcount()
    targets_full = ordered.loc[ordered["pos"] >= 1].copy()

    total_full_targets = len(targets_full)
    controlled_cold_targets = int(targets_full["is_controlled_cold"].fillna(False).sum())
    natural_cold_targets = int(targets_full["is_natural_cold"].fillna(False).sum())
    warm_trainable_targets = int(
        (
            targets_full["internal_item_id"].notna()
            & (~targets_full["is_controlled_cold"].fillna(False))
        ).sum()
    )
    post_holdout_attrition_targets = int(
        total_full_targets
        - controlled_cold_targets
        - natural_cold_targets
        - warm_trainable_targets
    )

    # Đối với baseline ID-based, mọi target rơi vào cold item đều có xác suất trúng = 0
    warm_retention_ratio = float(warm_test_examples_count / max(1, total_full_targets))

    return {
        "full_test_sessions_len_ge_2": int(len(valid_sessions)),
        "full_test_all_prefix_targets": int(total_full_targets),
        "warm_evaluated_targets": int(warm_test_examples_count),
        "controlled_cold_targets_in_test": controlled_cold_targets,
        "controlled_cold_target_share_pct": float(
            controlled_cold_targets * 100.0 / max(1, total_full_targets)
        ),
        "natural_cold_targets_in_test": natural_cold_targets,
        "natural_cold_target_share_pct": float(
            natural_cold_targets * 100.0 / max(1, total_full_targets)
        ),
        "other_unseen_or_filtered_targets": post_holdout_attrition_targets,
        "warm_target_coverage_in_full_test_pct": float(warm_retention_ratio * 100.0),
        f"warm_recall@{k}": float(warm_metrics.get(f"recall@{k}", 0.0)),
        f"warm_mrr@{k}": float(warm_metrics.get(f"mrr@{k}", 0.0)),
        f"warm_ndcg@{k}": float(warm_metrics.get(f"ndcg@{k}", 0.0)),
        f"effective_full_test_recall@{k}_with_cold_zero": float(
            warm_metrics.get(f"recall@{k}", 0.0) * warm_retention_ratio
        ),
        f"effective_full_test_mrr@{k}_with_cold_zero": float(
            warm_metrics.get(f"mrr@{k}", 0.0) * warm_retention_ratio
        ),
        f"effective_full_test_ndcg@{k}_with_cold_zero": float(
            warm_metrics.get(f"ndcg@{k}", 0.0) * warm_retention_ratio
        ),
    }
