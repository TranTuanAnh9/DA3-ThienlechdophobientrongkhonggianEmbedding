from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from btl_k03.popularity_metrics import (
    compute_arp,
    compute_catalog_coverage,
    compute_gini_coefficient,
    compute_ranking_metrics_from_predictions,
)


def test_catalog_coverage_hand_example():
    # Catalog có 10 item hợp lệ; danh sách gợi ý xuất hiện 4 item khác nhau (1, 2, 3, 5)
    recommended = [[1, 2, 1], [2, 3, 5]]
    cov = compute_catalog_coverage(recommended, catalog_size=10)
    assert np.isclose(cov, 0.4)


def test_gini_coefficient_uniform_and_concentrated():
    # Phân phối đều hoàn toàn trên 5 item -> Gini = 0
    uniform_counts = [10, 10, 10, 10, 10]
    assert np.isclose(compute_gini_coefficient(uniform_counts), 0.0)

    # Tập trung hoàn toàn vào 1 item trong catalog 4 item: [0, 0, 0, 100] -> Gini = (4 - 1)/4 = 0.75
    concentrated_counts = [0, 0, 0, 100]
    assert np.isclose(compute_gini_coefficient(concentrated_counts), 0.75)


def test_average_recommendation_popularity_hand_example():
    popularity_map = {101: 50, 102: 10, 103: 0}
    recommendations = [101, 101, 102, 103]
    # Trung bình = (50 + 50 + 10 + 0) / 4 = 27.5
    arp = compute_arp(recommendations, popularity_map, use_log1p=False)
    assert np.isclose(arp, 27.5)


def test_ranking_metrics_hand_example():
    # Query 1 trúng ở rank 1 (MRR=1, NDCG=1)
    # Query 2 trúng ở rank 2 (MRR=0.5, NDCG=1/log2(3))
    # Query 3 trượt (MRR=0, NDCG=0)
    preds = np.array([[10, 20, 30], [11, 22, 33], [12, 23, 34]])
    targets = np.array([10, 22, 99])
    metrics = compute_ranking_metrics_from_predictions(preds, targets, k=3)

    assert np.isclose(metrics["recall@3"], 2.0 / 3.0)
    assert np.isclose(metrics["mrr@3"], (1.0 + 0.5 + 0.0) / 3.0)
    assert np.isclose(metrics["ndcg@3"], (1.0 + 1.0 / np.log2(3.0) + 0.0) / 3.0)
