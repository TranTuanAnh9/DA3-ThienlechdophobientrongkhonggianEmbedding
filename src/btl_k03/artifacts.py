"""Read embeddings from an export or the authoritative saved checkpoint."""
from pathlib import Path

import numpy as np
import torch


def load_item_embeddings(run_dir: Path) -> np.ndarray:
    """Return item rows 1..N (padding excluded), as the old NPY export did."""
    run_dir = Path(run_dir)
    export = run_dir / "item_embeddings.npy"
    if export.exists():
        return np.load(export, allow_pickle=False)
    checkpoint = torch.load(run_dir / "checkpoint.pth", map_location="cpu", weights_only=False)
    weight = checkpoint["model_state_dict"]["item_embedding.weight"]
    return weight.detach().cpu().numpy()[1:].astype(np.float32, copy=True)
