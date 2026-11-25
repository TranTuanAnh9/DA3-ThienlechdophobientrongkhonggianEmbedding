"""Exact small-cloud Vietoris--Rips PH over F_2, with live Torch endpoints.

Vertices enter at zero; a simplex enters at its longest Euclidean edge. Boundary
reduction and diagram assignment are discrete, detached operations. Selected
birth/death edge lengths retain autograd away from filtration/assignment ties.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def vietoris_rips_persistence(points: torch.Tensor, max_dimension: int = 1) -> dict[int, torch.Tensor]:
    """H0/H1 diagrams [birth, death], including essential H0 = [0, inf].

    Triangles kill H1 classes; graph cycles are not reported as persistent holes.
    Full filtration, no cutoff, at most 32 vertices. Omit zero-length intervals.
    Unpaired triangles are outside the requested dimensions (no H2 claim).
    """
    if points.ndim != 2 or not 1 <= len(points) <= 32:
        raise ValueError("points must have shape [1..32, dimension]")
    if max_dimension not in (0, 1):
        raise ValueError("Only H0 and H1 are supported")
    if not points.is_floating_point() or not torch.isfinite(points).all():
        raise ValueError("points must be finite floating-point coordinates")
    distances = torch.cdist(points, points, compute_mode="donot_use_mm_for_euclid_dist")
    detached = distances.detach().cpu().numpy()
    simplices = []
    for size in range(1, max_dimension + 3):
        for vertices in combinations(range(len(points)), size):
            edge = max(combinations(vertices, 2), key=lambda e: detached[e]) if size > 1 else None
            value = float(detached[edge]) if edge is not None else 0.0
            simplices.append((value, size - 1, vertices, edge))
    simplices.sort(key=lambda s: (s[0], s[1], s[2]))
    index = {s[2]: i for i, s in enumerate(simplices)}
    reduced: dict[int, set[int]] = {}
    pivot_to_column: dict[int, int] = {}
    births, paired_births = set(), set()
    pairs: dict[int, list[torch.Tensor]] = {d: [] for d in range(max_dimension + 1)}
    zero = points.sum() * 0.0

    def endpoint(i):
        edge = simplices[i][3]
        return distances[edge] if edge is not None else zero

    for column, (_, dimension, vertices, _) in enumerate(simplices):
        boundary = {index[face] for face in combinations(vertices, len(vertices) - 1)} if dimension else set()
        while boundary and max(boundary) in pivot_to_column:
            boundary ^= reduced[pivot_to_column[max(boundary)]]
        if not boundary:
            births.add(column)
            continue
        pivot = max(boundary)
        reduced[column] = boundary
        pivot_to_column[pivot] = column
        paired_births.add(pivot)
        homology_dimension = simplices[pivot][1]
        if homology_dimension <= max_dimension and simplices[column][0] > simplices[pivot][0]:
            pairs[homology_dimension].append(torch.stack((endpoint(pivot), endpoint(column))))
    for birth in sorted(births - paired_births):
        dimension = simplices[birth][1]
        if dimension <= max_dimension:
            pairs[dimension].append(torch.stack((endpoint(birth), zero.new_tensor(float("inf")))))
    return {d: torch.stack(rows) if rows else points.new_empty((0, 2)) for d, rows in pairs.items()}


def finite_diagram(diagram: torch.Tensor) -> torch.Tensor:
    return diagram[torch.isfinite(diagram).all(dim=1)]


def diagram_wasserstein_squared(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Squared W2, Euclidean ground metric, optimal matching to the diagonal.

    Explicitly exclude essential bars. Unnormalised sum; caller may divide by
    cloud size. Empty diagrams are valid and do not receive artificial bars.
    """
    a, b = finite_diagram(a), finite_diagram(b)
    n, m = len(a), len(b)
    zero = a.sum() * 0.0 + b.sum() * 0.0
    if n == 0:
        return ((b[:, 1] - b[:, 0]).square() / 2).sum() + zero
    if m == 0:
        return ((a[:, 1] - a[:, 0]).square() / 2).sum() + zero
    cost = a.new_full((n + m, n + m), float("inf"))
    cost[:n, :m] = (a[:, None, :] - b[None, :, :]).square().sum(dim=-1)
    ai, bi = torch.arange(n, device=a.device), torch.arange(m, device=a.device)
    cost[ai, m + ai] = (a[:, 1] - a[:, 0]).square() / 2
    cost[n + bi, bi] = (b[:, 1] - b[:, 0]).square() / 2
    cost[n:, m:] = 0.0
    row, col = linear_sum_assignment(cost.detach().cpu().numpy().astype(np.float64))
    return cost[row, col].sum()
