"""Run controlled, measured PH experiments; never reuse legacy result paths."""
from pathlib import Path
import argparse
from datetime import datetime
import sys

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from btl_k03.topology.topo_trainer import METHODS, train_and_evaluate_topology_seed
from btl_k03.utils import load_experiment_config, load_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["gru4rec", "narm", "sasrec"], default=["gru4rec"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["baseline", "cosine", "ph_h0", "ph_h0_h1"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2024, 2025])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-limit", type=int, default=2048)
    parser.add_argument("--validation-limit", type=int, default=2048)
    parser.add_argument("--full-data", action="store_true", help="Ignore sample limits; set epochs explicitly for full training")
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluate test AFTER validation checkpoint selection")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/topology/role_gradient_pilot.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--init-existing", action="store_true", help="Fine-tune every method from the same model/seed baseline checkpoint; exploratory only")
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1:
        parser.error("threads and batch-size must be positive")
    torch.set_num_threads(args.threads)
    root = args.output or ROOT / "results/persistent_homology" / datetime.now().strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    for name in args.models:
        for seed in args.seeds:
            for method in args.methods:
                config = load_experiment_config(name)
                config["topology_params"] = load_yaml(args.config)["topology_params"]
                config["training"].update(train_batch_size=args.batch_size, eval_batch_size=args.batch_size)
                if args.learning_rate is not None:
                    if args.learning_rate <= 0:
                        parser.error("learning-rate must be positive")
                    config["training"]["learning_rate"] = args.learning_rate
                result = train_and_evaluate_topology_seed(seed, config=config, model_name=name, method=method,
                    max_epochs_override=args.epochs, device_str=args.device,
                    train_limit=None if args.full_data else args.train_limit,
                    validation_limit=None if args.full_data else args.validation_limit,
                    evaluate_test=args.evaluate_test, output_dir=root / name / method / f"seed_{seed}",
                    init_checkpoint=ROOT / "runs" / name / f"seed_{seed}" / "checkpoint.pth" if args.init_existing else None)
                for split in ("validation", "test"):
                    if split not in result:
                        continue
                    for group in result[split]["ranking_by_group"]:
                        rows.append({"model": name, "method": method, "seed": seed, "split": split,
                            "status": result["status"], "best_epoch": result["best_epoch"],
                            "seconds": result["seconds"], "frozen_drift": result["frozen_embedding_max_abs_drift"],
                            **group, **result[split]["popularity_bias"]})
                pd.DataFrame(rows).to_csv(root / "comparison.csv", index=False)
    print(f"Measured results: {root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
