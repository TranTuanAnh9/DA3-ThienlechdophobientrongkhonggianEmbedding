"""
Module Topological Regularization & Persistence Geometry for Session Recommendation.
Nghiên cứu: Hiệu chỉnh thiên lệch độ phổ biến trong không gian embedding bằng Tô-pô Lân cận Differentiable.
"""

from .persistence_loss import CosineTailAlignmentLoss, TopologicalPersistenceLoss
from .persistent_homology import vietoris_rips_persistence, diagram_wasserstein_squared
from .topo_eval import evaluate_topological_properties, compute_anisotropy_spectrum
from .topo_trainer import train_and_evaluate_topology_seed

__all__ = [
    "TopologicalPersistenceLoss",
    "CosineTailAlignmentLoss",
    "vietoris_rips_persistence",
    "diagram_wasserstein_squared",
    "evaluate_topological_properties",
    "compute_anisotropy_spectrum",
    "train_and_evaluate_topology_seed",
]
