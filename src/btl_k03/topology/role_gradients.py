"""Selective adjustment of the two CE paths through a tied embedding table.

PH supplies a local gradient direction, not a claim of semantic correctness.
The projection acts BEFORE Adam; it does not guarantee norm preservation or
loss descent after Adam preconditioning, momentum and clipping.
"""
from __future__ import annotations

import torch

ROLE_METHODS = ("role_sum", "role_input_scale", "role_projection", "role_tangent", "role_ph")


def combine_role_gradients(weight, grad_input, grad_output, warm_mask, frequencies,
                           method, strength=.5, tail_threshold=20, topology_gradient=None):
    if method not in ROLE_METHODS or not 0 <= strength <= 1:
        raise ValueError("Invalid role-gradient method or projection strength")
    gi, go = grad_input.detach(), grad_output.detach()
    active = gi.norm(dim=1) > 1e-12
    eligible = warm_mask & (frequencies > 0) & (frequencies < tail_threshold) & active
    conflict = (gi * go).sum(-1) < 0
    selected = torch.zeros_like(eligible)
    if method == "role_sum":
        adjusted = gi + go
    elif method == "role_input_scale":
        adjusted = (1 + strength) * gi + go
        selected = active & warm_mask
    else:
        unit = torch.nn.functional.normalize(weight.detach(), dim=-1)
        guide = gi if method != "role_ph" else topology_gradient
        if guide is None:
            raise ValueError("role_ph needs the live PH objective gradient")
        guide = guide.detach()
        output = go
        if method in ("role_tangent", "role_ph"):
            guide = guide - (guide * unit).sum(-1, keepdim=True) * unit
            output = go - (go * unit).sum(-1, keepdim=True) * unit
        dot = (output * guide).sum(-1, keepdim=True)
        square_norm = guide.square().sum(-1, keepdim=True)
        selected = eligible & conflict & (dot[:, 0] < 0) & (square_norm[:, 0] > 1e-24)
        projection = dot.clamp_max(0) / square_norm.clamp_min(1e-24) * guide
        adjusted = gi + go - strength * projection * selected[:, None]
    adjusted = adjusted.masked_fill(~warm_mask[:, None], 0)
    change = (adjusted - (gi + go)).norm()
    stats = {"input_active_items": int(active.sum()), "role_conflict_items": int((conflict & active & warm_mask).sum()),
             "adjusted_items": int(selected.sum()), "gradient_adjustment_norm": float(change),
             "relative_gradient_change": float(change / (gi + go).norm().clamp_min(1e-12))}
    return adjusted, stats


def backward_role_loss(ce_loss, weight, output_leaf, warm_mask, frequencies,
                       method, strength=.5, tail_threshold=20, topology_loss=None):
    """One CE forward/dropout realization; separate output leaf and input E.

    CE backward gives every encoder parameter its ordinary CE gradient. Only E
    is replaced. autograd.grad(PH, E) does not add a PH gradient to the encoder.
    Output-leaf gradients and input-E gradients sum exactly to tied CE gradient.
    """
    topology_gradient = None
    if method == "role_ph":
        if topology_loss is None:
            raise ValueError("Missing PH loss")
        topology_gradient = torch.autograd.grad(topology_loss, weight, retain_graph=True)[0]
    ce_loss.backward()
    adjusted, stats = combine_role_gradients(weight, weight.grad, output_leaf.grad,
        warm_mask, frequencies, method, strength, tail_threshold, topology_gradient)
    weight.grad = adjusted
    return stats
