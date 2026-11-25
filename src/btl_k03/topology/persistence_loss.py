"""Actual PH regularisation and an explicitly named cosine-only control."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .persistent_homology import diagram_wasserstein_squared, vietoris_rips_persistence


def session_dispersion(session_vectors: torch.Tensor) -> torch.Tensor:
    """Off-diagonal squared cosine; does not promise isotropy when B > D."""
    if len(session_vectors) < 2:
        return session_vectors.sum() * 0.0
    unit = F.normalize(session_vectors, dim=-1)
    gram = unit @ unit.T
    off_diagonal = ~torch.eye(len(unit), dtype=torch.bool, device=unit.device)
    return gram[off_diagonal].square().mean()


class CosineTailAlignmentLoss(nn.Module):
    """Pairwise angular hinge control. This is NOT persistent homology."""
    def __init__(self, tau=0.45, lambda_anisotropy=0.05, tail_threshold=20):
        super().__init__()
        self.tau = tau
        self.lambda_anisotropy = lambda_anisotropy
        self.tail_threshold = tail_threshold

    def forward(self, session_vectors, target_embeddings, item_frequencies, target_ids):
        frequency = item_frequencies[target_ids]
        tail = (frequency > 0) & (frequency < self.tail_threshold)
        distance = 1 - F.cosine_similarity(session_vectors, target_embeddings, dim=-1)
        alignment = F.relu(distance[tail] - self.tau).mean() if tail.any() else session_vectors.sum() * 0
        return alignment + self.lambda_anisotropy * session_dispersion(session_vectors)


class TopologicalPersistenceLoss(nn.Module):
    """Match local H0/H1 diagrams of sessions to detached target-item diagrams.

    Anchor clouds on rare TRAIN targets, selecting distinct target-ID neighbors
    by target-embedding distance in the same training batch. The paired cloud
    uses corresponding session vectors. L2-normalise both; filtration uses
    Euclidean chord distance [0, 2]. Reference/selection are detached, gradients
    flow through sessions, including their input embedding path.

    H0 preserves connectivity; H1 preserves actual loops killed by triangles.
    Long bars do NOT establish semantic quality. Tail accuracy is a hypothesis.
    """
    def __init__(self, neighbors=8, max_clouds=8, max_dimension=1, h1_weight=1.0,
                 lambda_anisotropy=0.05, tail_threshold=20):
        super().__init__()
        if not 3 <= neighbors <= 32 or max_clouds < 1:
            raise ValueError("neighbors must be 3..32 and max_clouds >= 1")
        if max_dimension not in (0, 1) or h1_weight < 0 or lambda_anisotropy < 0 or tail_threshold <= 0:
            raise ValueError("Invalid PH loss parameters")
        self.neighbors, self.max_clouds = neighbors, max_clouds
        self.max_dimension, self.h1_weight = max_dimension, h1_weight
        self.lambda_anisotropy, self.tail_threshold = lambda_anisotropy, tail_threshold
        self.last_stats: dict[str, float] = {}
        # Independent RNG: PH cloud sampling must not change subsequent dropout.
        self.sampling_generator = torch.Generator().manual_seed(2026)

    def forward(self, session_vectors, target_embeddings, item_frequencies, target_ids):
        if session_vectors.shape != target_embeddings.shape or len(target_ids) != len(session_vectors):
            raise ValueError("Sessions, targets and embeddings must have matching batch shapes")
        student = F.normalize(session_vectors.float(), dim=-1)
        reference = F.normalize(target_embeddings.detach().float(), dim=-1)
        frequencies = item_frequencies[target_ids]
        eligible = (frequencies > 0) & (frequencies < self.tail_threshold)
        anchors = eligible.nonzero().flatten()
        if len(anchors) > self.max_clouds:
            selection = torch.randperm(len(anchors), generator=self.sampling_generator)[:self.max_clouds]
            anchors = anchors[selection.to(anchors.device)]
        representatives = {}
        # Avoid per-element device synchronisation in the batch selection loop.
        target_list = target_ids.detach().cpu().tolist()
        frequency_list = frequencies.detach().cpu().tolist()
        for i, target in enumerate(target_list):
            if frequency_list[i] > 0:
                representatives.setdefault(target, i)
        losses0, losses1 = [], []
        bars1 = 0
        for anchor in anchors.tolist():
            candidates = [i for target, i in representatives.items() if target != target_list[anchor]]
            if len(candidates) < 2:
                continue
            candidate_ids = torch.tensor(candidates, device=student.device)
            distances = (reference[candidate_ids] - reference[anchor]).square().sum(-1)
            nearest = candidate_ids[torch.argsort(distances, stable=True)[:self.neighbors - 1]]
            ids = torch.cat((nearest.new_tensor([anchor]), nearest))
            predicted = vietoris_rips_persistence(student[ids], self.max_dimension)
            expected = vietoris_rips_persistence(reference[ids], self.max_dimension)
            # Normalise W2 sum by number of finite H0 bars; same scale for H1.
            losses0.append(diagram_wasserstein_squared(predicted[0], expected[0]) / (len(ids) - 1))
            if self.max_dimension == 1:
                losses1.append(diagram_wasserstein_squared(predicted[1], expected[1]) / (len(ids) - 1))
                bars1 += len(predicted[1]) + len(expected[1])
        zero = student.sum() * 0.0
        loss0 = torch.stack(losses0).mean() if losses0 else zero
        loss1 = torch.stack(losses1).mean() if losses1 else zero
        dispersion = session_dispersion(student)
        self.last_stats = {"ph_h0": float(loss0.detach()), "ph_h1": float(loss1.detach()),
                           "ph_clouds": len(losses0), "ph_h1_bars": bars1,
                           "dispersion": float(dispersion.detach())}
        return loss0 + self.h1_weight * loss1 + self.lambda_anisotropy * dispersion
