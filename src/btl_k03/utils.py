from __future__ import annotations

import copy
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .paths import CONFIGS_DIR, PROCESSED_DIGI_DIR


def set_global_seed(seed: int, deterministic_cudnn: bool = True) -> None:
    """Cố định toàn bộ random seed cho Python, NumPy và PyTorch (nếu có)."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if deterministic_cudnn and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return data or {}


def save_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, allow_unicode=True, sort_keys=False)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def load_experiment_config(model_name: str) -> dict[str, Any]:
    """Đọc và hợp nhất configs/data.yaml, configs/common.yaml và configs/{model_name}.yaml."""
    data_cfg = load_yaml(CONFIGS_DIR / "data.yaml")
    common_cfg = load_yaml(CONFIGS_DIR / "common.yaml")
    model_cfg = load_yaml(CONFIGS_DIR / f"{model_name.lower()}.yaml")

    merged = copy.deepcopy(common_cfg)
    merged["data_config"] = data_cfg
    merged["model_name"] = model_cfg.get("model_name", model_name.lower())
    merged["architecture"] = model_cfg.get("architecture", model_name.upper())

    model_params = copy.deepcopy(model_cfg.get("model_params", {}))
    model_params.setdefault("embedding_size", common_cfg.get("embedding_size", 64))
    merged["model_params"] = model_params

    training_cfg = copy.deepcopy(common_cfg.get("training", {}))
    training_cfg.update(model_cfg.get("training_overrides", {}))
    merged["training"] = training_cfg

    return merged


def load_item_mapping(processed_dir: Path = PROCESSED_DIGI_DIR) -> pd.DataFrame:
    """Đọc bảng ánh xạ item chuẩn và sắp xếp đúng theo internal_item_id tăng dần (1..N)."""
    mapping = pd.read_csv(processed_dir / "item_mapping.csv")
    mapping = mapping.sort_values("internal_item_id", kind="mergesort").reset_index(drop=True)
    return mapping


def create_run_logger(log_path: Path, logger_name: str) -> logging.Logger:
    """Tạo logger ghi đồng thời ra file training.log và console."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
