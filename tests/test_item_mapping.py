from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "diginetica"


def test_item_mapping_is_unique_and_reserves_zero_for_padding():
    mapping = pd.read_csv(DATA_DIR / "item_mapping.csv")

    assert mapping["item_id"].is_unique
    assert mapping["internal_item_id"].is_unique
    assert mapping["internal_item_id"].min() == 1
    assert mapping["internal_item_id"].max() == len(mapping)
    assert set(mapping["internal_item_id"]) == set(range(1, len(mapping) + 1))


def test_all_processed_event_ids_exist_in_mapping():
    mapping = pd.read_csv(DATA_DIR / "item_mapping.csv")
    known_raw_ids = set(mapping["item_id"])
    known_internal_ids = set(mapping["internal_item_id"])

    for split in ("train", "validation", "test"):
        events = pd.read_parquet(DATA_DIR / f"{split}_events.parquet")
        assert set(events["item_id"]).issubset(known_raw_ids)
        assert set(events["internal_item_id"]).issubset(known_internal_ids)


def test_category_mapping_is_complete_for_model_catalog():
    items = pd.read_csv(DATA_DIR / "item_mapping.csv")
    categories = pd.read_csv(DATA_DIR / "category_mapping.csv")

    assert not items["category_id"].isna().any()
    assert categories["category_id"].is_unique
    assert categories["internal_category_id"].is_unique
    assert set(items["category_id"]).issubset(set(categories["category_id"]))
