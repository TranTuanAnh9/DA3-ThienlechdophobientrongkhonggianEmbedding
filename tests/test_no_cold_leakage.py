import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "diginetica"


def _controlled_cold_ids() -> set[int]:
    cold = pd.read_csv(DATA_DIR / "cold_items.csv")
    return set(cold.loc[cold["cold_type"] == "controlled_option_b", "item_id"])


def test_controlled_cold_is_exactly_twenty_percent_of_pre_holdout_catalog():
    cold_ids = _controlled_cold_ids()
    statistics = json.loads(
        (DATA_DIR / "dataset_statistics.json").read_text(encoding="utf-8")
    )
    ratio = statistics["controlled_cold_ratio_requested"]
    pre_holdout_items = pd.read_csv(
        ROOT / "results" / "tables" / "preprocessing_cold_start_summary.csv"
    )
    pre_holdout_count = int(
        pre_holdout_items.loc[
            pre_holdout_items["metric"] == "pre_holdout_filtered_train_items", "value"
        ].iloc[0]
    )

    assert len(cold_ids) == round(ratio * pre_holdout_count)


def test_controlled_cold_never_appears_in_training_events_or_targets():
    cold_ids = _controlled_cold_ids()
    train_events = pd.read_parquet(DATA_DIR / "train_events.parquet")
    train_examples = pd.read_parquet(DATA_DIR / "train_examples.parquet")

    assert cold_ids.isdisjoint(set(train_events["item_id"]))
    assert cold_ids.isdisjoint(set(train_examples["target_item_id"]))


def test_main_validation_and_test_only_contain_final_warm_items():
    items = pd.read_csv(DATA_DIR / "item_mapping.csv")
    warm_ids = set(items.loc[~items["is_controlled_cold"], "item_id"])

    for split in ("validation", "test"):
        events = pd.read_parquet(DATA_DIR / f"{split}_events.parquet")
        examples = pd.read_parquet(DATA_DIR / f"{split}_examples.parquet")
        assert set(events["item_id"]).issubset(warm_ids)
        assert set(examples["target_item_id"]).issubset(warm_ids)


def test_every_controlled_cold_category_retains_a_warm_item():
    items = pd.read_csv(DATA_DIR / "item_mapping.csv")
    cold_categories = set(items.loc[items["is_controlled_cold"], "category_id"])
    warm_categories = set(items.loc[~items["is_controlled_cold"], "category_id"])

    assert cold_categories.issubset(warm_categories)
