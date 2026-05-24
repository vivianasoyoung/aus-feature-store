# CBA Feature Store

End-to-end ML feature store for banking fraud detection. Demonstrates the
**Feast + MLflow + FastAPI** pattern with a clean separation between
features and labels.

## How this fits into the broader project

This is one of four repos in the CBA portfolio:

| Repo | Role |
| --- | --- |
| `cba-banking-pipeline` | Generates and ingests raw transactions into Postgres. |
| `cba-dbt-analytics` | Transforms raw data into analytics marts. |
| `cba-fraud-streaming` | Rule-based real-time fraud detection. Produces the **labels** this repo uses. |
| **`cba-feature-store`** *(you are here)* | Computes ML features, trains a model, serves predictions. |

> The fact that **labels come from `cba-fraud-streaming`** (not from the same columns
> we use as features) is what makes the model genuinely predictive rather than
> trivially circular. See `compute_features.py` and `train_model.py`.

## Architecture

```
raw transactions (CSV from cba-banking-pipeline)
        +
flagged transactions (CSV exported from cba-fraud-streaming's Postgres)
        ↓
compute_features.py  ← labels JOINED from external signal
        ↓
Parquet (offline store)  →  feast apply / materialize  →  SQLite (online store)
        ↓                                                       ↓
train_model.py                                              fraud_api.py
   ↓                                                              ↓
MLflow Registry  ────────────────────────────────────────→  /score endpoint
```

## Setup

```bash
pip install -r requirements.txt

# 1. Compute features + join labels
python features/compute_features.py \
    --transactions ../cba-banking-pipeline/data/raw/transactions.csv \
    --flagged ../cba-fraud-streaming/data/flagged_transactions.csv \
    --out feature_repo/data/account_features.parquet

# 2. Apply Feast definitions + materialize to online store
cd feature_repo && feast apply && feast materialize-incremental $(date +%F) && cd ..

# 3. Train + register model
mlflow ui --port 5001 &
python training/train_model.py --features feature_repo/data/account_features.parquet

# 4. Serve
uvicorn serving.fraud_api:app --port 8001
```

## Expected metrics

With labels sourced from the streaming engine, hold-out AUC lands in the
**0.7–0.9** range depending on how the rule mix is tuned. If you see
AUC ≥ 0.99, suspect leakage and check that `is_fraud_account` is NOT a
function of the columns in `FEATURE_COLS`. There's a regression test for
this in `tests/test_compute_features.py`.

## What I'd improve in production

See `REAL_WORLD_NOTES.md` for the honest "this is a demo, here's what's
different at scale" discussion — useful prep for the inevitable interview
question.
