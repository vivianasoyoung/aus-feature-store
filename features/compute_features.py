"""
compute_features.py
-------------------
Computes fraud detection features from raw transaction data and joins
in fraud labels from the streaming pipeline's flagged_transactions table.

Labels come from an EXTERNAL signal (the streaming rule engine), not from
the same columns we use as features. This avoids target leakage.
"""

import argparse
import os
from datetime import timedelta

import pandas as pd


WINDOW_DAYS = 7


def compute_account_features(
    transactions_path: str,
    flagged_path: str,
    output_path: str,
) -> pd.DataFrame:
    df = pd.read_csv(transactions_path, parse_dates=["transaction_date"])
    debits = df[df["transaction_type"] == "DEBIT"].copy()

    # Restrict to the trailing window the feature names actually describe.
    max_date = debits["transaction_date"].max()
    window_start = max_date - timedelta(days=WINDOW_DAYS)
    window = debits[debits["transaction_date"] >= window_start].copy()

    window["hour"] = window["transaction_date"].dt.hour
    window["is_online"] = (window["channel"] == "ONLINE").astype(int)
    window["is_night"] = ((window["hour"] < 6) | (window["hour"] > 22)).astype(int)

    # Per-account observed-days denominator (avoids dividing by a hardcoded 180).
    days_per_account = (
        window.groupby("account_id")["transaction_date"]
        .agg(lambda s: max((s.max() - s.min()).days, 1))
        .rename("observed_days")
    )

    features = window.groupby("account_id").agg(
        transaction_count_7d=("transaction_id", "count"),
        total_spend_7d=("amount", "sum"),
        avg_transaction_value=("amount", "mean"),
        max_transaction_value=("amount", "max"),
        unique_categories=("merchant_category", "nunique"),
        online_transaction_ratio=("is_online", "mean"),
        night_transaction_ratio=("is_night", "mean"),
    )
    features = features.join(days_per_account)
    features["avg_daily_spend"] = features["total_spend_7d"] / features["observed_days"]
    features = features.drop(columns=["observed_days"]).reset_index()

    # Labels from an INDEPENDENT signal: the streaming fraud engine.
    # Any account with ≥1 flagged transaction in the window is positive.
    flagged = pd.read_csv(flagged_path, parse_dates=["flagged_at"])
    flagged_in_window = flagged[flagged["flagged_at"] >= window_start]
    positive_accounts = set(flagged_in_window["account_id"].unique())
    features["is_fraud_account"] = features["account_id"].isin(positive_accounts).astype(int)

    # Per-account event timestamp = latest observation for that account.
    last_seen = window.groupby("account_id")["transaction_date"].max()
    features["event_timestamp"] = features["account_id"].map(last_seen)

    float_cols = [
        "total_spend_7d",
        "avg_transaction_value",
        "max_transaction_value",
        "online_transaction_ratio",
        "night_transaction_ratio",
        "avg_daily_spend",
    ]
    features[float_cols] = features[float_cols].round(4)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features.to_parquet(output_path, index=False)

    pos_rate = features["is_fraud_account"].mean()
    print(f"Wrote {len(features):,} rows → {output_path}")
    print(f"Positive label rate: {pos_rate:.2%}")
    return features


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--transactions", required=True, help="Path to raw transactions CSV")
    p.add_argument("--flagged", required=True, help="Path to flagged_transactions CSV exported from fraud DB")
    p.add_argument("--out", required=True, help="Output parquet path")
    args = p.parse_args()
    compute_account_features(args.transactions, args.flagged, args.out)


if __name__ == "__main__":
    main()
