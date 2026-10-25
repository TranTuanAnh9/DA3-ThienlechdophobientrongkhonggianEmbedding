from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def compute_embedding_norms(embeddings: np.ndarray) -> np.ndarray:
    """Tính chuẩn L2 (Euclidean norm) cho từng hàng vector item embedding."""
    matrix = np.asarray(embeddings, dtype=np.float64)
    return np.linalg.norm(matrix, axis=1)


def compute_category_centroid_metrics(
    embeddings: np.ndarray,
    item_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """
    Tính tâm category (centroid) chỉ từ các WARM item trong category đó (vì cold item chưa được học),
    sau đó đo khoảng cách Euclidean và độ tương đồng Cosine từ mọi item (cả warm và cold)
    đến tâm category của chính nó.
    """
    emb = np.asarray(embeddings, dtype=np.float64)
    if len(emb) != len(item_mapping):
        raise ValueError(
            f"Số hàng embedding ({len(emb)}) không khớp số dòng item_mapping ({len(item_mapping)})."
        )

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    unit_emb = emb / np.clip(norms, 1e-12, None)

    df = item_mapping.copy().reset_index(drop=True)
    df["embedding_norm"] = norms.ravel()

    warm_mask = df["is_trainable_warm"].to_numpy(dtype=bool)
    categories = df["category_id"].to_numpy()

    # Tính centroid của từng category dựa trên các warm item thuộc category đó
    unique_cats = np.unique(categories)
    cat_centroids: dict[int, np.ndarray] = {}
    cat_unit_centroids: dict[int, np.ndarray] = {}
    cat_warm_counts: dict[int, int] = {}

    for cat_id in unique_cats:
        cat_warm_idx = np.where((categories == cat_id) & warm_mask)[0]
        cat_warm_counts[int(cat_id)] = int(len(cat_warm_idx))
        if len(cat_warm_idx) > 0:
            centroid = emb[cat_warm_idx].mean(axis=0)
            c_norm = float(np.linalg.norm(centroid))
            unit_centroid = centroid / max(c_norm, 1e-12)
            cat_centroids[int(cat_id)] = centroid
            cat_unit_centroids[int(cat_id)] = unit_centroid

    distances = np.full(len(df), np.nan, dtype=np.float64)
    cosines = np.full(len(df), np.nan, dtype=np.float64)
    warm_peers = np.zeros(len(df), dtype=np.int64)

    for idx, cat_id in enumerate(categories):
        cid = int(cat_id)
        warm_peers[idx] = cat_warm_counts.get(cid, 0)
        if cid in cat_centroids:
            distances[idx] = float(np.linalg.norm(emb[idx] - cat_centroids[cid]))
            cosines[idx] = float(np.dot(unit_emb[idx], cat_unit_centroids[cid]))

    df["category_warm_peers"] = warm_peers
    df["distance_to_category_centroid"] = distances
    df["cosine_to_category_centroid"] = cosines
    return df


def build_item_level_analysis_table(
    model_name: str,
    seed: int,
    embeddings: np.ndarray,
    item_rec_catalog: pd.DataFrame,
) -> pd.DataFrame:
    """
    Tạo bảng phân tích trung tâm cấp độ item (Mục 11 tài liệu tổng quan đề tài),
    kết nối tần suất train, nhóm popularity, embedding norm, khoảng cách tâm category
    với số lần xuất hiện trong top-20 (rec_count) và exposure_share.
    """
    ordered_catalog = item_rec_catalog.sort_values(
        "internal_item_id", kind="mergesort"
    ).reset_index(drop=True)
    with_geometry = compute_category_centroid_metrics(embeddings, ordered_catalog)

    result = pd.DataFrame(
        {
            "model": model_name.lower(),
            "seed": int(seed),
            "internal_item_id": with_geometry["internal_item_id"].astype(np.int64),
            "item_id": with_geometry["item_id"].astype(np.int64),
            "category_id": with_geometry["category_id"].astype(np.int64),
            "internal_category_id": with_geometry["internal_category_id"].astype(np.int64),
            "train_freq": with_geometry["train_frequency"].astype(np.int64),
            "popularity_group": with_geometry["popularity_group"].astype(str),
            "is_cold": with_geometry["is_controlled_cold"].astype(bool),
            "is_trainable_warm": with_geometry["is_trainable_warm"].astype(bool),
            "candidate_for_main_evaluation": with_geometry[
                "candidate_for_main_evaluation"
            ].astype(bool),
            "embedding_norm": with_geometry["embedding_norm"].astype(np.float64),
            "rec_count": with_geometry["rec_count"].astype(np.int64),
            "exposure_share": with_geometry["exposure_share"].astype(np.float64),
            "distance_to_category_centroid": with_geometry[
                "distance_to_category_centroid"
            ].astype(np.float64),
            "cosine_to_category_centroid": with_geometry[
                "cosine_to_category_centroid"
            ].astype(np.float64),
            "category_warm_peers": with_geometry["category_warm_peers"].astype(np.int64),
        }
    )
    return result


def spearman_with_bootstrap_ci(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 2024,
) -> dict[str, float]:
    """
    Tính hệ số tương quan hạng Spearman kèm p-value và khoảng tin cậy Bootstrap 95%
    (theo đúng yêu cầu Mục 10.2 & Mục 15 của đề tài).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]

    n = len(x_arr)
    if n < 3 or np.all(x_arr == x_arr[0]) or np.all(y_arr == y_arr[0]):
        return {
            "n": int(n),
            "rho": 0.0,
            "p_value": 1.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
        }

    res = stats.spearmanr(x_arr, y_arr)
    rho = float(res.statistic)
    p_val = float(res.pvalue)

    if n_bootstrap > 0:
        rng = np.random.default_rng(seed)
        rx = stats.rankdata(x_arr)
        ry = stats.rankdata(y_arr)
        boot_rhos = np.empty(n_bootstrap, dtype=np.float64)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            bx = rx[idx]
            by = ry[idx]
            std_x = bx.std()
            std_y = by.std()
            if std_x > 0 and std_y > 0:
                boot_rhos[b] = np.corrcoef(bx, by)[0, 1]
            else:
                boot_rhos[b] = 0.0
        alpha = (1.0 - confidence) / 2.0
        ci_low = float(np.quantile(boot_rhos, alpha))
        ci_high = float(np.quantile(boot_rhos, 1.0 - alpha))
    else:
        ci_low, ci_high = rho, rho

    return {
        "n": int(n),
        "rho": rho,
        "p_value": p_val,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def analyze_three_core_relationships(
    item_table: pd.DataFrame,
    n_bootstrap: int = 500,
    seed: int = 2024,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Phân tích 3 mối quan hệ cốt lõi trên tập warm items (Mục 12.1 & 12.2 tài liệu đề tài):
    1. train_freq <-> embedding_norm
    2. train_freq <-> rec_count
    3. embedding_norm <-> rec_count
    Kèm phân tích tách theo từng bin tần suất (rare, medium, popular) và hồi quy tuyến tính
    kiểm soát biến nhiễu log(1 + train_freq).
    """
    warm = item_table.loc[item_table["is_trainable_warm"]].copy()
    all_items = item_table.copy()

    core_rows = []
    # 1. Warm items: 3 core relationships
    pairs = [
        ("warm_only", "train_freq", "embedding_norm", warm["train_freq"], warm["embedding_norm"]),
        ("warm_only", "train_freq", "rec_count", warm["train_freq"], warm["rec_count"]),
        ("warm_only", "embedding_norm", "rec_count", warm["embedding_norm"], warm["rec_count"]),
        ("all_catalog_incl_cold", "train_freq", "embedding_norm", all_items["train_freq"], all_items["embedding_norm"]),
    ]

    for scope, var_x, var_y, s_x, s_y in pairs:
        stat_dict = spearman_with_bootstrap_ci(
            s_x, s_y, n_bootstrap=n_bootstrap, seed=seed
        )
        core_rows.append(
            {
                "model": str(item_table["model"].iloc[0]),
                "seed": int(item_table["seed"].iloc[0]),
                "scope": scope,
                "pair": f"{var_x} <-> {var_y}",
                "var_x": var_x,
                "var_y": var_y,
                **stat_dict,
            }
        )

    # 2. Within-bin correlation: embedding_norm <-> rec_count inside each popularity_group
    bin_rows = []
    for grp in ("rare", "medium", "popular"):
        sub = warm.loc[warm["popularity_group"] == grp]
        stat_norm_rec = spearman_with_bootstrap_ci(
            sub["embedding_norm"], sub["rec_count"], n_bootstrap=n_bootstrap, seed=seed
        )
        stat_freq_norm = spearman_with_bootstrap_ci(
            sub["train_freq"], sub["embedding_norm"], n_bootstrap=n_bootstrap, seed=seed
        )
        bin_rows.append(
            {
                "model": str(item_table["model"].iloc[0]),
                "seed": int(item_table["seed"].iloc[0]),
                "popularity_group": grp,
                "n_items": int(len(sub)),
                "norm_rec_rho": stat_norm_rec["rho"],
                "norm_rec_p_value": stat_norm_rec["p_value"],
                "norm_rec_ci_low": stat_norm_rec["ci_low"],
                "norm_rec_ci_high": stat_norm_rec["ci_high"],
                "freq_norm_rho": stat_freq_norm["rho"],
                "freq_norm_p_value": stat_freq_norm["p_value"],
            }
        )

    # 3. Multiple OLS regression: log(1 + rec_count) = beta_0 + beta_1 * log(1 + train_freq) + beta_2 * norm
    y_reg = np.log1p(warm["rec_count"].to_numpy(dtype=np.float64))
    x_freq = np.log1p(warm["train_freq"].to_numpy(dtype=np.float64))
    x_norm = warm["embedding_norm"].to_numpy(dtype=np.float64)

    # Chuẩn hóa Z-score để có thêm hệ số beta chuẩn hóa (standardized beta)
    x_design = np.column_stack([np.ones_like(x_freq), x_freq, x_norm])
    coeffs, _, _, _ = np.linalg.lstsq(x_design, y_reg, rcond=None)
    y_pred = x_design @ coeffs
    residuals = y_reg - y_pred
    dof = max(1, len(y_reg) - 3)
    mse = float(np.sum(residuals**2) / dof)
    cov_beta = mse * np.linalg.pinv(x_design.T @ x_design)
    se_beta = np.sqrt(np.clip(np.diag(cov_beta), 1e-15, None))
    t_stats = coeffs / se_beta
    p_vals = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stats), df=dof))
    ss_tot = float(np.sum((y_reg - y_reg.mean()) ** 2))
    r_squared = float(1.0 - np.sum(residuals**2) / max(ss_tot, 1e-12))

    regression_summary = {
        "model": str(item_table["model"].iloc[0]),
        "seed": int(item_table["seed"].iloc[0]),
        "n_warm_items": int(len(warm)),
        "intercept_beta0": float(coeffs[0]),
        "log1p_freq_beta1": float(coeffs[1]),
        "log1p_freq_se": float(se_beta[1]),
        "log1p_freq_t": float(t_stats[1]),
        "log1p_freq_p": float(p_vals[1]),
        "norm_beta2": float(coeffs[2]),
        "norm_se": float(se_beta[2]),
        "norm_t": float(t_stats[2]),
        "norm_p": float(p_vals[2]),
        "r_squared": r_squared,
    }

    return pd.DataFrame(core_rows), pd.DataFrame(bin_rows), regression_summary


def sample_category_cosine_pairs(
    embeddings: np.ndarray,
    item_mapping: pd.DataFrame,
    n_pairs_per_group: int = 50000,
    seed: int = 2024,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Lấy mẫu ngẫu nhiên có kiểm soát (cố định seed) các cặp item để tính Cosine Similarity
    trên 4 nhóm:
    1. warm_same_category: hai item warm cùng category
    2. warm_diff_category: hai item warm khác category
    3. cold_warm_same_category: một item cold và một item warm cùng category
    4. cold_warm_diff_category: một item cold và một item warm khác category
    Tránh tạo ma trận 37,583 x 37,583 (~1.4 tỷ cặp) theo đúng hướng dẫn Mục 10.4.
    """
    rng = np.random.default_rng(seed)
    emb = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    unit_emb = emb / np.clip(norms, 1e-12, None)

    df = item_mapping.sort_values("internal_item_id", kind="mergesort").reset_index(drop=True)
    cats = df["category_id"].to_numpy()
    is_warm = df["is_trainable_warm"].to_numpy(dtype=bool)
    is_cold = df["is_controlled_cold"].to_numpy(dtype=bool)

    warm_indices = np.where(is_warm)[0]
    cold_indices = np.where(is_cold)[0]

    # Gom index theo category
    warm_by_cat: dict[int, np.ndarray] = {}
    cold_by_cat: dict[int, np.ndarray] = {}
    for c in np.unique(cats):
        cid = int(c)
        w_idx = np.where(is_warm & (cats == c))[0]
        c_idx = np.where(is_cold & (cats == c))[0]
        if len(w_idx) >= 2:
            warm_by_cat[cid] = w_idx
        if len(c_idx) >= 1 and len(w_idx) >= 1:
            cold_by_cat[cid] = (c_idx, w_idx)

    # 1. Warm - Warm cùng category
    valid_warm_cats = list(warm_by_cat.keys())
    cat_weights = np.array([len(warm_by_cat[c]) for c in valid_warm_cats], dtype=np.float64)
    cat_probs = cat_weights / cat_weights.sum()

    chosen_cats = rng.choice(valid_warm_cats, size=n_pairs_per_group, p=cat_probs)
    idx_a_same = np.empty(n_pairs_per_group, dtype=np.int64)
    idx_b_same = np.empty(n_pairs_per_group, dtype=np.int64)
    for i, cid in enumerate(chosen_cats):
        pool = warm_by_cat[int(cid)]
        pair = rng.choice(pool, size=2, replace=False)
        idx_a_same[i], idx_b_same[i] = pair[0], pair[1]
    cos_warm_same = np.sum(unit_emb[idx_a_same] * unit_emb[idx_b_same], axis=1)

    # 2. Warm - Warm khác category
    idx_a_diff = np.empty(n_pairs_per_group, dtype=np.int64)
    idx_b_diff = np.empty(n_pairs_per_group, dtype=np.int64)
    filled = 0
    while filled < n_pairs_per_group:
        batch = min(n_pairs_per_group - filled + 2048, n_pairs_per_group * 2)
        cand_a = rng.choice(warm_indices, size=batch, replace=True)
        cand_b = rng.choice(warm_indices, size=batch, replace=True)
        valid = cats[cand_a] != cats[cand_b]
        cand_a, cand_b = cand_a[valid], cand_b[valid]
        take = min(len(cand_a), n_pairs_per_group - filled)
        idx_a_diff[filled : filled + take] = cand_a[:take]
        idx_b_diff[filled : filled + take] = cand_b[:take]
        filled += take
    cos_warm_diff = np.sum(unit_emb[idx_a_diff] * unit_emb[idx_b_diff], axis=1)

    # 3. Cold - Warm cùng category
    valid_cw_cats = list(cold_by_cat.keys())
    cw_weights = np.array([len(cold_by_cat[c][0]) for c in valid_cw_cats], dtype=np.float64)
    cw_probs = cw_weights / cw_weights.sum()
    chosen_cw_cats = rng.choice(valid_cw_cats, size=n_pairs_per_group, p=cw_probs)
    idx_c_same = np.empty(n_pairs_per_group, dtype=np.int64)
    idx_w_same = np.empty(n_pairs_per_group, dtype=np.int64)
    for i, cid in enumerate(chosen_cw_cats):
        c_pool, w_pool = cold_by_cat[int(cid)]
        idx_c_same[i] = rng.choice(c_pool)
        idx_w_same[i] = rng.choice(w_pool)
    cos_cw_same = np.sum(unit_emb[idx_c_same] * unit_emb[idx_w_same], axis=1)

    # 4. Cold - Warm khác category
    idx_c_diff = np.empty(n_pairs_per_group, dtype=np.int64)
    idx_w_diff = np.empty(n_pairs_per_group, dtype=np.int64)
    filled = 0
    while filled < n_pairs_per_group:
        batch = min(n_pairs_per_group - filled + 2048, n_pairs_per_group * 2)
        cand_c = rng.choice(cold_indices, size=batch, replace=True)
        cand_w = rng.choice(warm_indices, size=batch, replace=True)
        valid = cats[cand_c] != cats[cand_w]
        cand_c, cand_w = cand_c[valid], cand_w[valid]
        take = min(len(cand_c), n_pairs_per_group - filled)
        idx_c_diff[filled : filled + take] = cand_c[:take]
        idx_w_diff[filled : filled + take] = cand_w[:take]
        filled += take
    cos_cw_diff = np.sum(unit_emb[idx_c_diff] * unit_emb[idx_w_diff], axis=1)

    groups_data = [
        ("warm_same_category", cos_warm_same),
        ("warm_diff_category", cos_warm_diff),
        ("cold_warm_same_category", cos_cw_same),
        ("cold_warm_diff_category", cos_cw_diff),
    ]

    summary_rows = []
    for name, arr in groups_data:
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1))
        se_val = std_val / np.sqrt(len(arr))
        summary_rows.append(
            {
                "pair_group": name,
                "n_pairs": int(len(arr)),
                "mean_cosine": mean_val,
                "std_cosine": std_val,
                "median_cosine": float(np.median(arr)),
                "ci95_low": float(mean_val - 1.96 * se_val),
                "ci95_high": float(mean_val + 1.96 * se_val),
            }
        )

    u_warm, p_warm = stats.mannwhitneyu(cos_warm_same, cos_warm_diff, alternative="two-sided")
    t_warm, tp_warm = stats.ttest_ind(cos_warm_same, cos_warm_diff, equal_var=False)

    u_cold, p_cold = stats.mannwhitneyu(cos_cw_same, cos_cw_diff, alternative="two-sided")

    tests_dict = {
        "warm_same_vs_diff_mean_gap": float(np.mean(cos_warm_same) - np.mean(cos_warm_diff)),
        "warm_same_vs_diff_ttest_t": float(t_warm),
        "warm_same_vs_diff_ttest_p": float(tp_warm),
        "warm_same_vs_diff_mannwhitney_u": float(u_warm),
        "warm_same_vs_diff_mannwhitney_p": float(p_warm),
        "cold_warm_same_vs_diff_mean_gap": float(np.mean(cos_cw_same) - np.mean(cos_cw_diff)),
        "cold_warm_same_vs_diff_mannwhitney_u": float(u_cold),
        "cold_warm_same_vs_diff_mannwhitney_p": float(p_cold),
    }

    return pd.DataFrame(summary_rows), tests_dict


def compute_tsne_sample(
    embeddings: np.ndarray,
    item_table: pd.DataFrame,
    sample_size: int = 8000,
    perplexity: int = 30,
    seed: int = 2024,
    n_top_categories: int = 8,
) -> pd.DataFrame:
    """
    Lấy mẫu phân tầng 5,000 - 10,000 item qua 4 nhóm (popular, medium, rare, cold)
    và chạy t-SNE giảm chiều xuống 2D (kèm PCA tiền xử lý nếu chiều > 32).
    """
    rng = np.random.default_rng(seed)
    df = item_table.sort_values("internal_item_id", kind="mergesort").reset_index(drop=True)
    emb = np.asarray(embeddings, dtype=np.float32)

    groups = ["popular", "medium", "rare", "cold"]
    per_group = sample_size // len(groups)
    sampled_indices = []

    for grp in groups:
        grp_idx = np.where(df["popularity_group"].to_numpy() == grp)[0]
        take = min(len(grp_idx), per_group)
        chosen = rng.choice(grp_idx, size=take, replace=False)
        sampled_indices.append(chosen)

    sampled_idx = np.sort(np.concatenate(sampled_indices))
    sub_df = df.iloc[sampled_idx].copy().reset_index(drop=True)
    sub_emb = emb[sampled_idx]

    if sub_emb.shape[1] > 32:
        pca = PCA(n_components=32, random_state=seed)
        sub_emb_input = pca.fit_transform(sub_emb)
    else:
        sub_emb_input = sub_emb

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=seed,
    )
    coords = tsne.fit_transform(sub_emb_input)

    sub_df["tsne_1"] = coords[:, 0]
    sub_df["tsne_2"] = coords[:, 1]
    sub_df["tsne_perplexity"] = int(perplexity)
    sub_df["tsne_seed"] = int(seed)

    # Đánh dấu các category lớn nhất để tô màu biểu đồ t-SNE theo category dễ quan sát
    top_cats = (
        df.loc[df["is_trainable_warm"], "category_id"]
        .value_counts()
        .head(n_top_categories)
        .index.tolist()
    )
    sub_df["category_display"] = sub_df["category_id"].apply(
        lambda c: f"Cat {int(c)}" if c in top_cats else "Other Categories"
    )

    return sub_df
