from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
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
    for opt_col in (
        "category_source",
        "has_regular_category",
        "ever_special_offer",
        "special_offer_click_share",
    ):
        if opt_col in with_geometry.columns:
            result[opt_col] = with_geometry[opt_col].to_numpy()
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

    # Hồi quy trên thang đo gốc; sau đó quy đổi sang standardized beta để có thể
    # so sánh công bằng giữa các mô hình có thang norm khác nhau.
    x_design = np.column_stack([np.ones_like(x_freq), x_freq, x_norm])
    coeffs, _, _, _ = np.linalg.lstsq(x_design, y_reg, rcond=None)
    y_pred = x_design @ coeffs
    residuals = y_reg - y_pred
    dof = max(1, len(y_reg) - 3)
    mse = float(np.sum(residuals**2) / dof)
    cov_beta = mse * np.linalg.pinv(x_design.T @ x_design)
    se_beta = np.sqrt(np.clip(np.diag(cov_beta), 1e-15, None))
    t_stats = coeffs / se_beta
    # survival function ổn định số học hơn 1 - CDF khi |t| rất lớn.
    p_vals = 2.0 * stats.t.sf(np.abs(t_stats), df=dof)
    ss_tot = float(np.sum((y_reg - y_reg.mean()) ** 2))
    r_squared = float(1.0 - np.sum(residuals**2) / max(ss_tot, 1e-12))
    adjusted_r_squared = float(
        1.0 - (1.0 - r_squared) * (len(y_reg) - 1) / max(1, len(y_reg) - 3)
    )
    rmse = float(np.sqrt(np.mean(residuals**2)))

    y_sd = float(np.std(y_reg, ddof=1))
    freq_sd = float(np.std(x_freq, ddof=1))
    norm_sd = float(np.std(x_norm, ddof=1))
    freq_scale = freq_sd / max(y_sd, 1e-12)
    norm_scale = norm_sd / max(y_sd, 1e-12)
    freq_beta_std = float(coeffs[1] * freq_scale)
    norm_beta_std = float(coeffs[2] * norm_scale)
    freq_se_std = float(se_beta[1] * freq_scale)
    norm_se_std = float(se_beta[2] * norm_scale)
    t_critical = float(stats.t.ppf(0.975, df=dof))

    regression_summary = {
        "model": str(item_table["model"].iloc[0]),
        "seed": int(item_table["seed"].iloc[0]),
        "n_warm_items": int(len(warm)),
        "intercept_beta0": float(coeffs[0]),
        "log1p_freq_beta1": float(coeffs[1]),
        "log1p_freq_se": float(se_beta[1]),
        "log1p_freq_t": float(t_stats[1]),
        "log1p_freq_p": float(p_vals[1]),
        "log1p_freq_ci95_low": float(coeffs[1] - t_critical * se_beta[1]),
        "log1p_freq_ci95_high": float(coeffs[1] + t_critical * se_beta[1]),
        "log1p_freq_beta1_standardized": freq_beta_std,
        "log1p_freq_standardized_se": freq_se_std,
        "log1p_freq_standardized_ci95_low": float(freq_beta_std - t_critical * freq_se_std),
        "log1p_freq_standardized_ci95_high": float(freq_beta_std + t_critical * freq_se_std),
        "norm_beta2": float(coeffs[2]),
        "norm_se": float(se_beta[2]),
        "norm_t": float(t_stats[2]),
        "norm_p": float(p_vals[2]),
        "norm_ci95_low": float(coeffs[2] - t_critical * se_beta[2]),
        "norm_ci95_high": float(coeffs[2] + t_critical * se_beta[2]),
        "norm_beta2_standardized": norm_beta_std,
        "norm_standardized_se": norm_se_std,
        "norm_standardized_ci95_low": float(norm_beta_std - t_critical * norm_se_std),
        "norm_standardized_ci95_high": float(norm_beta_std + t_critical * norm_se_std),
        "r_squared": r_squared,
        "adjusted_r_squared": adjusted_r_squared,
        "rmse": rmse,
        "zero_rec_count_share": float(np.mean(warm["rec_count"].to_numpy() == 0)),
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
    max_iter: int = 1000,
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
        max_iter=max_iter,
        random_state=seed,
    )
    coords = tsne.fit_transform(sub_emb_input)

    sub_df["tsne_1"] = coords[:, 0]
    sub_df["tsne_2"] = coords[:, 1]
    sub_df["tsne_perplexity"] = int(perplexity)
    sub_df["tsne_seed"] = int(seed)
    sub_df["tsne_max_iter"] = int(max_iter)

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


def build_query_and_item_accuracy_tables(
    test_examples: pd.DataFrame,
    top20: pd.DataFrame,
    topk: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ghép mỗi truy vấn test với target thật và danh sách Top-K để tạo:

    1. Bảng cấp truy vấn: target, nhóm popularity và hit@K.
    2. Bảng cấp item: số lần làm target, số lần hit và item-level Recall@K.

    ``query_id`` trong ``top20`` phải tương ứng với thứ tự dòng của
    ``test_examples``. Hàm kiểm tra chặt số truy vấn, số recommendation trên mỗi
    truy vấn và ``session_id`` để tránh ghép nhầm target với recommendation.
    """
    required_examples = {"session_id", "target_internal_item_id"}
    required_topk = {"session_id", "query_id", "rank", "internal_item_id"}
    missing_examples = sorted(required_examples.difference(test_examples.columns))
    missing_topk = sorted(required_topk.difference(top20.columns))
    if missing_examples:
        raise ValueError(f"test_examples thiếu cột: {missing_examples}")
    if missing_topk:
        raise ValueError(f"top20 thiếu cột: {missing_topk}")

    queries = test_examples.reset_index(drop=True).copy()
    queries.insert(0, "query_id", np.arange(len(queries), dtype=np.int64))
    ranked = top20.sort_values(["query_id", "rank"], kind="mergesort").reset_index(drop=True)

    query_ids = ranked["query_id"].to_numpy(dtype=np.int64)
    if len(query_ids) == 0 or query_ids.min() != 0 or query_ids.max() != len(queries) - 1:
        raise ValueError("query_id trong top20 không phủ đúng toàn bộ test_examples.")
    counts = ranked.groupby("query_id", sort=True).size()
    if len(counts) != len(queries) or not bool((counts == int(topk)).all()):
        raise ValueError(f"Mỗi query phải có đúng {topk} recommendation.")
    if ranked.duplicated(["query_id", "internal_item_id"]).any():
        raise ValueError("Một query chứa item recommendation bị lặp.")

    ranked_sessions = ranked.groupby("query_id", sort=True)["session_id"].first().to_numpy()
    query_sessions = queries["session_id"].to_numpy()
    if not np.array_equal(ranked_sessions, query_sessions):
        raise ValueError("session_id giữa top20 và test_examples không khớp theo query_id.")

    targets = queries["target_internal_item_id"].to_numpy(dtype=np.int64)
    recommended = ranked["internal_item_id"].to_numpy(dtype=np.int64)
    row_hits = recommended == targets[query_ids]
    hit_counts_by_query = np.bincount(
        query_ids,
        weights=row_hits.astype(np.int64),
        minlength=len(queries),
    ).astype(np.int64)
    if int(hit_counts_by_query.max(initial=0)) > 1:
        raise ValueError("Một target xuất hiện nhiều hơn một lần trong Top-K của cùng query.")
    queries[f"hit@{topk}"] = hit_counts_by_query.astype(bool)

    item_accuracy = (
        queries.groupby("target_internal_item_id", sort=True)
        .agg(
            target_count=("query_id", "size"),
            hit_count=(f"hit@{topk}", "sum"),
        )
        .reset_index()
        .rename(columns={"target_internal_item_id": "internal_item_id"})
    )
    item_accuracy["target_count"] = item_accuracy["target_count"].astype(np.int64)
    item_accuracy["hit_count"] = item_accuracy["hit_count"].astype(np.int64)
    item_accuracy[f"item_recall@{topk}"] = (
        item_accuracy["hit_count"] / item_accuracy["target_count"]
    )
    return queries, item_accuracy


def attach_item_accuracy(
    item_table: pd.DataFrame,
    item_accuracy: pd.DataFrame,
    topk: int = 20,
) -> pd.DataFrame:
    """Ghép target/hit cấp item vào bảng geometry–exposure hiện có."""
    if item_table["internal_item_id"].duplicated().any():
        raise ValueError("item_table có internal_item_id bị trùng.")
    if item_accuracy["internal_item_id"].duplicated().any():
        raise ValueError("item_accuracy có internal_item_id bị trùng.")
    out = item_table.merge(item_accuracy, on="internal_item_id", how="left", validate="one_to_one")
    out["target_count"] = out["target_count"].fillna(0).astype(np.int64)
    out["hit_count"] = out["hit_count"].fillna(0).astype(np.int64)
    recall_col = f"item_recall@{topk}"
    out[recall_col] = np.where(
        out["target_count"] > 0,
        out["hit_count"] / out["target_count"],
        np.nan,
    )
    return out


def summarize_geometry_and_outcomes_by_group(
    item_table: pd.DataFrame,
    topk: int = 20,
) -> pd.DataFrame:
    """
    Đặt geometry, exposure và accuracy cạnh nhau cho popular/medium/rare.

    Recall theo nhóm được tính từ tổng hit / tổng target (micro average), đúng
    cùng định nghĩa với đánh giá accuracy của notebook huấn luyện.
    """
    recall_col = f"item_recall@{topk}"
    required = {
        "model", "seed", "is_trainable_warm", "popularity_group", "train_freq",
        "embedding_norm", "distance_to_category_centroid", "cosine_to_category_centroid",
        "rec_count", "target_count", "hit_count", recall_col,
    }
    missing = sorted(required.difference(item_table.columns))
    if missing:
        raise ValueError(f"item_table thiếu cột để tổng hợp geometry–outcome: {missing}")

    warm = item_table.loc[item_table["is_trainable_warm"]].copy()
    total_slots = int(warm["rec_count"].sum())
    rows: list[dict[str, Any]] = []
    for group in ("popular", "medium", "rare"):
        sub = warm.loc[warm["popularity_group"] == group]
        target_count = int(sub["target_count"].sum())
        hit_count = int(sub["hit_count"].sum())
        rows.append({
            "model": str(warm["model"].iloc[0]),
            "seed": int(warm["seed"].iloc[0]),
            "popularity_group": group,
            "item_count": int(len(sub)),
            "train_frequency_mean": float(sub["train_freq"].mean()),
            "embedding_norm_mean": float(sub["embedding_norm"].mean()),
            "centroid_distance_mean": float(sub["distance_to_category_centroid"].mean()),
            "centroid_cosine_mean": float(sub["cosine_to_category_centroid"].mean()),
            "recommendation_count": int(sub["rec_count"].sum()),
            "exposure_share_pct": 100.0 * float(sub["rec_count"].sum()) / max(total_slots, 1),
            "within_group_coverage_pct": 100.0 * float((sub["rec_count"] > 0).mean()),
            "target_count": target_count,
            "hit_count": hit_count,
            f"recall@{topk}": float(hit_count / target_count) if target_count else np.nan,
        })
    return pd.DataFrame(rows)


def compute_geometry_output_correlations(
    item_table: pd.DataFrame,
    topk: int = 20,
) -> pd.DataFrame:
    """
    Tính Spearman giữa ba đặc trưng geometry và hai outcome:
    recommendation count (exposure) và item-level Recall@K (accuracy).

    Kết quả được tính trên toàn bộ warm catalog và riêng từng popularity group.
    Tương quan accuracy chỉ dùng item thực sự xuất hiện làm target trong test.
    """
    geometry_vars = (
        "embedding_norm",
        "distance_to_category_centroid",
        "cosine_to_category_centroid",
    )
    recall_col = f"item_recall@{topk}"
    warm = item_table.loc[item_table["is_trainable_warm"]].copy()
    scopes: list[tuple[str, pd.DataFrame]] = [("all_warm", warm)]
    scopes.extend(
        (group, warm.loc[warm["popularity_group"] == group])
        for group in ("rare", "medium", "popular")
    )

    rows: list[dict[str, Any]] = []
    for scope, scope_df in scopes:
        for geometry_var in geometry_vars:
            for outcome, require_target in (("rec_count", False), (recall_col, True)):
                sub = scope_df.loc[scope_df["target_count"] > 0].copy() if require_target else scope_df
                x = sub[geometry_var].to_numpy(dtype=np.float64)
                y = sub[outcome].to_numpy(dtype=np.float64)
                mask = np.isfinite(x) & np.isfinite(y)
                x, y = x[mask], y[mask]
                if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
                    rho, p_value = np.nan, np.nan
                else:
                    rho, p_value = stats.spearmanr(x, y)
                rows.append({
                    "model": str(warm["model"].iloc[0]),
                    "seed": int(warm["seed"].iloc[0]),
                    "scope": scope,
                    "geometry_variable": geometry_var,
                    "outcome": outcome,
                    "n_items": int(len(x)),
                    "rho": float(rho),
                    "p_value": float(p_value),
                })
    return pd.DataFrame(rows)


def _standardize_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    sd = float(np.std(values, ddof=1))
    if not np.isfinite(sd) or sd <= 1e-12:
        raise ValueError("Không thể chuẩn hóa biến có độ lệch chuẩn bằng 0.")
    return (values - float(np.mean(values))) / sd


def _fit_standardized_ols(y: np.ndarray, predictors: dict[str, np.ndarray]) -> dict[str, Any]:
    names = list(predictors)
    y_std = _standardize_vector(np.asarray(y, dtype=np.float64))
    x_cols = [_standardize_vector(np.asarray(predictors[name], dtype=np.float64)) for name in names]
    x_design = np.column_stack([np.ones(len(y_std)), *x_cols])
    coeffs, _, _, _ = np.linalg.lstsq(x_design, y_std, rcond=None)
    y_pred = x_design @ coeffs
    residuals = y_std - y_pred
    dof = max(1, len(y_std) - x_design.shape[1])
    mse = float(np.sum(residuals**2) / dof)
    cov_beta = mse * np.linalg.pinv(x_design.T @ x_design)
    se = np.sqrt(np.clip(np.diag(cov_beta), 1e-15, None))
    t_stats = coeffs / se
    p_values = 2.0 * stats.t.sf(np.abs(t_stats), df=dof)
    t_critical = float(stats.t.ppf(0.975, df=dof))
    ss_tot = float(np.sum((y_std - y_std.mean()) ** 2))
    r_squared = float(1.0 - np.sum(residuals**2) / max(ss_tot, 1e-12))
    out: dict[str, Any] = {"r_squared": r_squared, "n": int(len(y_std))}
    for idx, name in enumerate(names, start=1):
        out[f"{name}_beta_std"] = float(coeffs[idx])
        out[f"{name}_p_value"] = float(p_values[idx])
        out[f"{name}_ci95_low"] = float(coeffs[idx] - t_critical * se[idx])
        out[f"{name}_ci95_high"] = float(coeffs[idx] + t_critical * se[idx])
    return out


def fit_geometry_exposure_models(item_table: pd.DataFrame) -> pd.DataFrame:
    """
    Hồi quy exposure cấp item, luôn kiểm soát log(1 + train frequency).

    Chạy các mô hình riêng cho norm, centroid distance và centroid cosine để
    tránh diễn giải sai khi các đặc trưng geometry phụ thuộc lẫn nhau.
    """
    warm = item_table.loc[item_table["is_trainable_warm"]].copy()
    y = np.log1p(warm["rec_count"].to_numpy(dtype=np.float64))
    freq = np.log1p(warm["train_freq"].to_numpy(dtype=np.float64))
    geometry_vars = (
        "embedding_norm",
        "distance_to_category_centroid",
        "cosine_to_category_centroid",
    )
    baseline = _fit_standardized_ols(y, {"log1p_train_frequency": freq})
    rows = [{
        "model": str(warm["model"].iloc[0]),
        "seed": int(warm["seed"].iloc[0]),
        "model_spec": "frequency_only",
        "geometry_variable": "none",
        "frequency_beta_std": baseline["log1p_train_frequency_beta_std"],
        "geometry_beta_std": np.nan,
        "geometry_p_value": np.nan,
        "geometry_ci95_low": np.nan,
        "geometry_ci95_high": np.nan,
        "r_squared": baseline["r_squared"],
        "delta_r_squared_vs_frequency": 0.0,
        "n_items": baseline["n"],
    }]
    for geometry_var in geometry_vars:
        fit = _fit_standardized_ols(
            y,
            {
                "log1p_train_frequency": freq,
                geometry_var: warm[geometry_var].to_numpy(dtype=np.float64),
            },
        )
        rows.append({
            "model": str(warm["model"].iloc[0]),
            "seed": int(warm["seed"].iloc[0]),
            "model_spec": f"frequency_plus_{geometry_var}",
            "geometry_variable": geometry_var,
            "frequency_beta_std": fit["log1p_train_frequency_beta_std"],
            "geometry_beta_std": fit[f"{geometry_var}_beta_std"],
            "geometry_p_value": fit[f"{geometry_var}_p_value"],
            "geometry_ci95_low": fit[f"{geometry_var}_ci95_low"],
            "geometry_ci95_high": fit[f"{geometry_var}_ci95_high"],
            "r_squared": fit["r_squared"],
            "delta_r_squared_vs_frequency": fit["r_squared"] - baseline["r_squared"],
            "n_items": fit["n"],
        })
    return pd.DataFrame(rows)


def fit_geometry_accuracy_models(
    query_table: pd.DataFrame,
    item_table: pd.DataFrame,
    topk: int = 20,
    random_state: int = 2024,
) -> pd.DataFrame:
    """
    Logistic regression cấp truy vấn cho xác suất hit@K.

    Mỗi mô hình mở rộng thêm đúng một biến geometry bên cạnh train frequency.
    Hệ số là standardized log-odds; ``delta_log_loss_vs_frequency`` dương nghĩa
    là biến geometry cải thiện dự đoán hit so với mô hình chỉ có frequency.
    """
    feature_cols = [
        "internal_item_id", "train_freq", "embedding_norm",
        "distance_to_category_centroid", "cosine_to_category_centroid",
    ]
    feature_table = item_table.loc[item_table["is_trainable_warm"], feature_cols].copy()
    queries = query_table.merge(
        feature_table,
        left_on="target_internal_item_id",
        right_on="internal_item_id",
        how="left",
        validate="many_to_one",
    )
    if queries[feature_cols[1:]].isna().any().any():
        raise ValueError("Không ghép được geometry cho một số warm target query.")
    y = queries[f"hit@{topk}"].astype(int).to_numpy()
    if np.unique(y).size != 2:
        raise ValueError("Biến hit phải chứa cả 0 và 1 để chạy logistic regression.")

    freq = _standardize_vector(np.log1p(queries["train_freq"].to_numpy(dtype=np.float64)))
    geometry_vars = (
        "embedding_norm",
        "distance_to_category_centroid",
        "cosine_to_category_centroid",
    )
    null_prob = float(np.mean(y))
    null_predictions = np.full(len(y), null_prob, dtype=np.float64)
    null_log_loss = float(log_loss(y, null_predictions, labels=[0, 1]))

    def fit_logistic(columns: list[np.ndarray]) -> tuple[np.ndarray, float, float]:
        x = np.column_stack(columns)
        estimator = LogisticRegression(
            C=1e6,
            solver="lbfgs",
            max_iter=2000,
            random_state=int(random_state),
        )
        estimator.fit(x, y)
        probabilities = estimator.predict_proba(x)[:, 1]
        model_log_loss = float(log_loss(y, probabilities, labels=[0, 1]))
        mcfadden_r2 = float(1.0 - model_log_loss / max(null_log_loss, 1e-12))
        return estimator.coef_[0].astype(np.float64), model_log_loss, mcfadden_r2

    baseline_coef, baseline_loss, baseline_r2 = fit_logistic([freq])
    rows = [{
        "model": str(item_table["model"].iloc[0]),
        "seed": int(item_table["seed"].iloc[0]),
        "model_spec": "frequency_only",
        "geometry_variable": "none",
        "frequency_beta_std": float(baseline_coef[0]),
        "geometry_beta_std": np.nan,
        "log_loss": baseline_loss,
        "delta_log_loss_vs_frequency": 0.0,
        "mcfadden_r_squared": baseline_r2,
        "delta_mcfadden_r_squared_vs_frequency": 0.0,
        "n_queries": int(len(queries)),
    }]
    for geometry_var in geometry_vars:
        geometry = _standardize_vector(queries[geometry_var].to_numpy(dtype=np.float64))
        coeffs, model_loss, pseudo_r2 = fit_logistic([freq, geometry])
        rows.append({
            "model": str(item_table["model"].iloc[0]),
            "seed": int(item_table["seed"].iloc[0]),
            "model_spec": f"frequency_plus_{geometry_var}",
            "geometry_variable": geometry_var,
            "frequency_beta_std": float(coeffs[0]),
            "geometry_beta_std": float(coeffs[1]),
            "log_loss": model_loss,
            "delta_log_loss_vs_frequency": baseline_loss - model_loss,
            "mcfadden_r_squared": pseudo_r2,
            "delta_mcfadden_r_squared_vs_frequency": pseudo_r2 - baseline_r2,
            "n_queries": int(len(queries)),
        })
    return pd.DataFrame(rows)
