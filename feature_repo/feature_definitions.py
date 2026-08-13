"""
feature_definitions.py
-----------------------
Defines banking fraud detection features for the AU banking feature store.
These features are computed from transaction history and used by
fraud detection ML models for both training and real-time serving.
"""

from datetime import timedelta
import pandas as pd
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64, String

# ── Entity ───────────────────────────────────────────────────────────────────
# An entity is the thing we're computing features about — in this case accounts
account = Entity(
    name="account_id",
    description="An Australian retail bank account"
)

# ── Data Source ───────────────────────────────────────────────────────────────
# Points to the precomputed features parquet file
account_features_source = FileSource(
    path="data/account_features.parquet",
    timestamp_field="event_timestamp",
)

# ── Feature View ──────────────────────────────────────────────────────────────
# A feature view is a group of related features computed from a data source
account_transaction_features = FeatureView(
    name="account_transaction_features",
    entities=[account],
    ttl=timedelta(days=7),
    schema=[
        Field(name="transaction_count_7d",    dtype=Int64,   description="Number of transactions in last 7 days"),
        Field(name="total_spend_7d",          dtype=Float64, description="Total spend in last 7 days"),
        Field(name="avg_transaction_value",   dtype=Float64, description="Average transaction value"),
        Field(name="max_transaction_value",   dtype=Float64, description="Maximum single transaction value"),
        Field(name="unique_categories",       dtype=Int64,   description="Number of unique merchant categories"),
        Field(name="online_transaction_ratio",dtype=Float64, description="Ratio of online to total transactions"),
        Field(name="night_transaction_ratio", dtype=Float64, description="Ratio of night-time transactions"),
        Field(name="avg_daily_spend",         dtype=Float64, description="Average daily spend"),
        Field(name="is_high_risk",            dtype=Int64,   description="1 if account shows high risk patterns"),
    ],
    source=account_features_source,
)
