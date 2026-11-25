import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform

def compute_anisotropy_spectrum(item_embeddings: np.ndarray) -> dict:
    """
    Đo độ mất đẳng hướng (Anisotropy) của không gian embedding.
    Nếu singular values phân rã quá nhanh -> Không gian bị bẹp/nón (Anisotropic).
    Nếu singular values dàn đều -> Không gian đẳng hướng hoàn hảo (Isotropic).
    """
    # Trừ vector trung bình (Centering)
    mean_vec = np.mean(item_embeddings, axis=0, keepdims=True)
    centered = item_embeddings - mean_vec
    
    # SVD
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    variance_explained = (s ** 2) / np.sum(s ** 2)
    
    # Anisotropy ratio = singular value lớn nhất / tổng singular values
    top1_ratio = variance_explained[0]
    top5_ratio = np.sum(variance_explained[:5])
    
    return {
        "top1_singular_ratio": float(top1_ratio),
        "top5_singular_ratio": float(top5_ratio),
        "singular_values": s[:10].tolist()
    }

def evaluate_topological_properties(embeddings: np.ndarray, item_frequencies: np.ndarray, tail_threshold: int = 20) -> dict:
    """
    Đánh giá tô-pô lân cận giữa nhóm Head và nhóm Tail (Item A vs Item B).
    """
    tail_indices = np.where((item_frequencies < tail_threshold) & (item_frequencies > 0))[0]
    head_indices = np.where(item_frequencies >= tail_threshold)[0]
    
    anisotropy_metrics = compute_anisotropy_spectrum(embeddings)
    
    # Tính L2 norm phân bố giữa Head và Tail
    tail_norms = np.linalg.norm(embeddings[tail_indices], axis=1) if len(tail_indices) > 0 else np.array([0])
    head_norms = np.linalg.norm(embeddings[head_indices], axis=1) if len(head_indices) > 0 else np.array([0])
    
    return {
        "anisotropy": anisotropy_metrics,
        "tail_mean_norm": float(np.mean(tail_norms)),
        "head_mean_norm": float(np.mean(head_norms)),
        "norm_ratio_head_to_tail": float(np.mean(head_norms) / (np.mean(tail_norms) + 1e-8)),
    }
