"""Inference-time popularity scoring control; this is not a novel PH method."""
from pathlib import Path
import json
import math

import torch
from torch import nn


class PopularityScoreCalibrator(nn.Module):
    """Apply a validation-selected, frozen coefficient to full-catalog logits.

    Model parameters/embeddings stay unchanged. The buffer adds one float per
    catalog entry. Candidate masking remains the caller's responsibility.
    """
    def __init__(self, train_frequencies, coefficient):
        super().__init__()
        frequency = torch.as_tensor(train_frequencies, dtype=torch.float32).clone()
        coefficient = float(coefficient)
        if frequency.ndim != 1 or not torch.isfinite(frequency).all() or (frequency < 0).any():
            raise ValueError("Expected finite nonnegative train frequencies")
        if not math.isfinite(coefficient) or coefficient < 0:
            raise ValueError("Expected a finite nonnegative coefficient")
        self.register_buffer("log_popularity_penalty", torch.log1p(frequency))
        self.coefficient = coefficient

    def forward(self, logits):
        if logits.shape[-1] != len(self.log_popularity_penalty):
            raise ValueError("Logits and training catalog sizes differ")
        return logits - self.coefficient * self.log_popularity_penalty.to(device=logits.device, dtype=logits.dtype)

    @classmethod
    def from_validation_policy(cls, policy_file: Path, item_mapping):
        policy = json.loads(Path(policy_file).read_text(encoding="utf-8"))
        if policy.get("selected_on") != "validation" or policy.get("test_seen_during_selection") is not False:
            raise ValueError("Expected a policy locked on validation before test evaluation")
        size = int(item_mapping.internal_item_id.max()) + 1
        frequency = torch.zeros(size)
        ids = torch.tensor(item_mapping.internal_item_id.to_numpy().copy(), dtype=torch.long)
        frequency[ids] = torch.tensor(item_mapping.train_frequency.to_numpy().copy(), dtype=torch.float32)
        return cls(frequency, policy["coefficient"])
