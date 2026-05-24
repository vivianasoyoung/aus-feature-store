"""
fraud_api.py
------------
FastAPI endpoint serving fraud risk scores. Backed by:
  - Feast online store for feature retrieval (account_id lookup)
  - MLflow Model Registry for the scoring model (versioned)
"""

import os
from datetime import datetime, timezone

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from feast import FeatureStore
from pydantic import BaseModel

FEATURE_REPO_PATH = os.getenv("FEATURE_REPO_PATH", "feature_repo")
MODEL_URI         = os.getenv("MODEL_URI", "models:/cba_fraud_model/Staging")
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))

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

app = FastAPI(title="CBA Fraud Detection API", version="2.0.0")

store: FeatureStore | None = None
model = None


@app.on_event("startup")
def _startup():
    global store, model
    store = FeatureStore(repo_path=FEATURE_REPO_PATH)
    model = mlflow.sklearn.load_model(MODEL_URI)


class Transaction(BaseModel):
    transaction_id: str
    account_id:     str
    amount:         float
    merchant_category: str
    channel:        str


class FraudScore(BaseModel):
    transaction_id: str
    account_id:     str
    risk_score:     float
    is_fraud:       bool
    explanation:    str
    scored_at:      str


@app.get("/health")
def health():
    return {"status": "ok", "model_uri": MODEL_URI, "threshold": DECISION_THRESHOLD}


@app.post("/score", response_model=FraudScore)
def score(txn: Transaction) -> FraudScore:
    feature_refs = [f"account_transaction_features:{c}" for c in FEATURE_COLS]
    response = store.get_online_features(
        features=feature_refs,
        entity_rows=[{"account_id": txn.account_id}],
    ).to_dict()

    if any(response[c][0] is None for c in FEATURE_COLS):
        # Cold-start account — flag for manual review with a conservative score.
        return FraudScore(
            transaction_id=txn.transaction_id,
            account_id=txn.account_id,
            risk_score=0.5,
            is_fraud=False,
            explanation="Unknown account — no features in online store",
            scored_at=datetime.now(timezone.utc).isoformat(),
        )

    X = pd.DataFrame({c: response[c] for c in FEATURE_COLS})
    prob = float(model.predict_proba(X)[0, 1])

    return FraudScore(
        transaction_id=txn.transaction_id,
        account_id=txn.account_id,
        risk_score=round(prob, 4),
        is_fraud=prob >= DECISION_THRESHOLD,
        explanation=f"model={MODEL_URI} threshold={DECISION_THRESHOLD}",
        scored_at=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/account/{account_id}")
def account_features(account_id: str):
    feature_refs = [f"account_transaction_features:{c}" for c in FEATURE_COLS]
    response = store.get_online_features(
        features=feature_refs,
        entity_rows=[{"account_id": account_id}],
    ).to_dict()
    if all(response[c][0] is None for c in FEATURE_COLS):
        raise HTTPException(404, f"No features for account {account_id}")
    return {"account_id": account_id, "features": {c: response[c][0] for c in FEATURE_COLS}}
