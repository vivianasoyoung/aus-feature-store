"""
Tests for the fraud scoring API.

The app depends on two external systems at runtime — a Feast online store and
a model pulled from the MLflow registry. Neither is available in CI, and
neither is what these tests are checking: the contract under test is the
serving logic sitting between them. So both are replaced with fakes and
`fraud_api.store` / `fraud_api.model` are set directly, which also means the
startup hook never fires.

The case worth guarding hardest is cold start. An account with no features in
the online store must NOT reach the model — a model asked to score all-null
features will happily return a number, and that number is meaningless. The
API returns a conservative 0.5 and says why instead.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from serving import fraud_api
from serving.fraud_api import FEATURE_COLS, app

KNOWN_ACCOUNT = "ACC0000001"

# Plausible feature values for a known account, in FEATURE_COLS order.
KNOWN_FEATURES = {
    "transaction_count_7d":     12,
    "total_spend_7d":           1843.55,
    "avg_transaction_value":    153.63,
    "max_transaction_value":    899.00,
    "unique_categories":        5,
    "online_transaction_ratio": 0.42,
    "night_transaction_ratio":  0.08,
    "avg_daily_spend":          263.36,
}

VALID_TXN = {
    "transaction_id":    "TXN0000001",
    "account_id":        KNOWN_ACCOUNT,
    "amount":            250.00,
    "merchant_category": "GROCERIES",
    "channel":           "EFTPOS",
}


class FakeOnlineResponse:
    def __init__(self, values: dict):
        self._values = values

    def to_dict(self):
        return {c: [self._values.get(c)] for c in FEATURE_COLS}


class FakeStore:
    """Returns features for KNOWN_ACCOUNT, nulls for anything else."""

    def __init__(self):
        self.calls = []

    def get_online_features(self, features, entity_rows):
        self.calls.append(entity_rows)
        account_id = entity_rows[0]["account_id"]
        if account_id == KNOWN_ACCOUNT:
            return FakeOnlineResponse(KNOWN_FEATURES)
        return FakeOnlineResponse({})


class FakeModel:
    """
    Returns a fixed fraud probability and records whether it was called.

    Returns a numpy array rather than a list because the app indexes the
    result as `[0, 1]` — sklearn's tuple-style indexing, which a plain list
    doesn't support. The fake has to match the real interface, not just the
    shape of the data.
    """

    def __init__(self, prob: float = 0.15):
        self.prob = prob
        self.calls = []

    def predict_proba(self, X):
        self.calls.append(X)
        return np.array([[1 - self.prob, self.prob]])


@pytest.fixture
def store():
    fake = FakeStore()
    fraud_api.store = fake
    return fake


@pytest.fixture
def model():
    fake = FakeModel()
    fraud_api.model = fake
    return fake


@pytest.fixture
def client(store, model):
    # Plain TestClient (not the context manager) so the startup hook, which
    # would try to reach Feast and MLflow for real, never runs.
    return TestClient(app)


def test_health_reports_model_and_threshold(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model_uri" in body
    assert "threshold" in body


def test_score_returns_model_probability_for_known_account(client, model):
    model.prob = 0.15
    r = client.post("/score", json=VALID_TXN)
    assert r.status_code == 200
    body = r.json()
    assert body["transaction_id"] == VALID_TXN["transaction_id"]
    assert body["account_id"] == KNOWN_ACCOUNT
    assert body["risk_score"] == pytest.approx(0.15)
    assert body["is_fraud"] is False
    assert body["scored_at"].endswith("+00:00")


def test_score_flags_fraud_at_or_above_threshold(client, model, monkeypatch):
    monkeypatch.setattr(fraud_api, "DECISION_THRESHOLD", 0.5)
    model.prob = 0.5  # exactly on the boundary — must flag, not pass
    r = client.post("/score", json=VALID_TXN)
    assert r.json()["is_fraud"] is True


def test_score_below_threshold_is_not_flagged(client, model, monkeypatch):
    monkeypatch.setattr(fraud_api, "DECISION_THRESHOLD", 0.5)
    model.prob = 0.4999
    r = client.post("/score", json=VALID_TXN)
    assert r.json()["is_fraud"] is False


def test_cold_start_account_never_reaches_the_model(client, model):
    """
    The important one. A model handed all-null features returns a number, and
    that number is noise. Unknown accounts must short-circuit before scoring.
    """
    txn = {**VALID_TXN, "account_id": "ACC9999999"}
    r = client.post("/score", json=txn)

    assert r.status_code == 200
    body = r.json()
    assert body["risk_score"] == 0.5
    assert body["is_fraud"] is False
    assert "Unknown account" in body["explanation"]
    assert model.calls == [], "model was called for an account with no features"


def test_score_rejects_payload_missing_required_field(client):
    incomplete = {k: v for k, v in VALID_TXN.items() if k != "amount"}
    assert client.post("/score", json=incomplete).status_code == 422


def test_score_rejects_non_numeric_amount(client):
    assert client.post("/score", json={**VALID_TXN, "amount": "lots"}).status_code == 422


def test_account_features_returns_all_feature_columns(client):
    r = client.get(f"/account/{KNOWN_ACCOUNT}")
    assert r.status_code == 200
    body = r.json()
    assert body["account_id"] == KNOWN_ACCOUNT
    assert set(body["features"]) == set(FEATURE_COLS)
    assert body["features"]["transaction_count_7d"] == 12


def test_account_features_404s_for_unknown_account(client):
    r = client.get("/account/ACC9999999")
    assert r.status_code == 404
    assert "ACC9999999" in r.json()["detail"]


def test_store_is_queried_with_the_requested_account(client, store):
    client.post("/score", json=VALID_TXN)
    assert store.calls == [[{"account_id": KNOWN_ACCOUNT}]]