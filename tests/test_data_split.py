from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "diginetica"


def _events(split: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / f"{split}_events.parquet")


def test_sessions_are_disjoint_between_splits():
    train_sessions = set(_events("train")["session_id"].unique())
    validation_sessions = set(_events("validation")["session_id"].unique())
    test_sessions = set(_events("test")["session_id"].unique())

    assert train_sessions.isdisjoint(validation_sessions)
    assert train_sessions.isdisjoint(test_sessions)
    assert validation_sessions.isdisjoint(test_sessions)


def test_every_final_session_has_at_least_two_events():
    for split in ("train", "validation", "test"):
        lengths = _events(split).groupby("session_id").size()
        assert lengths.min() >= 2


def test_events_are_ordered_within_each_session():
    for split in ("train", "validation", "test"):
        events = _events(split)
        ordered = events.sort_values(
            ["session_id", "timeframe", "row_order"], kind="mergesort"
        )
        assert events.index.equals(ordered.index)


def test_temporal_boundaries_follow_session_last_date():
    train = _events("train")
    validation = _events("validation")
    test = _events("test")

    assert train["last_event_date"].max() < pd.Timestamp("2016-05-19")
    assert validation["last_event_date"].min() >= pd.Timestamp("2016-05-19")
    assert validation["last_event_date"].max() < pd.Timestamp("2016-05-26")
    assert test["last_event_date"].min() >= pd.Timestamp("2016-05-26")
