"""
Smoke tests for compute_features.

The most important assertion is `test_label_is_not_a_pure_function_of_features`
— the previous version of the code had target leakage that made AUC=1.00.
This test guards against that regressing.
"""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from features.compute_features import compute_account_features

FEATURE_COLS = [
    "transaction_count_7d",
    "total_spend_7d",
    "avg_transaction_value",
    "max_transaction_value",
    "unique_categories",
    "online_transaction_ratio",
    "night_transaction_ratio",
    "avg_daily_spend",
]


@pytest.fixture
def fixture_dir(tmp_path: Path) -> Path:
    now = datetime(2025, 1, 30, 12, 0)

    txns = []
    for i, acc in enumerate(["ACC_A", "ACC_B", "ACC_C"]):
        for d in range(10):
            txns.append({
                "transaction_id": f"{acc}-{d}",
                "account_id": acc,
                "transaction_date": now - timedelta(days=d),
                "amount": 100 + i * 50,
                "transaction_type": "DEBIT",
                "channel": "ONLINE" if d % 2 == 0 else "EFTPOS",
                "merchant_category": "Groceries",
            })
    pd.DataFrame(txns).to_csv(tmp_path / "txns.csv", index=False)

    # Only ACC_A is "flagged" by the streaming engine
    pd.DataFrame([{
        "account_id": "ACC_A",
        "flagged_at": now - timedelta(days=1),
    }]).to_csv(tmp_path / "flagged.csv", index=False)

    return tmp_path


def test_features_have_expected_shape(fixture_dir: Path):
    out = fixture_dir / "out.parquet"
    df = compute_account_features(
        str(fixture_dir / "txns.csv"),
        str(fixture_dir / "flagged.csv"),
        str(out),
    )
    assert set(FEATURE_COLS).issubset(df.columns)
    assert "is_fraud_account" in df.columns
    assert "event_timestamp" in df.columns
    assert len(df) == 3


def test_label_comes_from_external_signal(fixture_dir: Path):
    """ACC_A is flagged; B and C are not. Label must reflect that, NOT be
    derived from the same feature columns the model trains on."""
    out = fixture_dir / "out.parquet"
    df = compute_account_features(
        str(fixture_dir / "txns.csv"),
        str(fixture_dir / "flagged.csv"),
        str(out),
    )
    labels = df.set_index("account_id")["is_fraud_account"]
    assert labels["ACC_A"] == 1
    assert labels["ACC_B"] == 0
    assert labels["ACC_C"] == 0


def test_event_timestamp_is_per_account_not_now(fixture_dir: Path):
    out = fixture_dir / "out.parquet"
    df = compute_account_features(
        str(fixture_dir / "txns.csv"),
        str(fixture_dir / "flagged.csv"),
        str(out),
    )
    # All accounts have the same most-recent date in the fixture
    assert df["event_timestamp"].nunique() == 1
    # …but it's the txn date, not "now"
    assert df["event_timestamp"].iloc[0] < pd.Timestamp.now()


def test_label_is_not_a_pure_function_of_features(fixture_dir: Path):
    """
    Regression guard for target leakage. If the label were a deterministic
    function of the feature columns, a model could trivially recover it.
    This test asserts no single feature, thresholded at its max, equals the
    label — the simplest form of leakage that produced the original AUC=1.00.
    """
    out = fixture_dir / "out.parquet"
    df = compute_account_features(
        str(fixture_dir / "txns.csv"),
        str(fixture_dir / "flagged.csv"),
        str(out),
    )
    # Assert the label is not identical to any single feature thresholded at
    # its max. If it were, the model could trivially recover it (leakage).
    for col in FEATURE_COLS:
        identical = (df["is_fraud_account"] == (df[col] == df[col].max())).all()
        assert not identical, f"Label is identical to {col} threshold — leakage!"
