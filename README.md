# Australian Banking Feature Store

End-to-end ML feature store for banking fraud detection. Demonstrates the **Feast + MLflow + FastAPI** pattern with a clean separation between features and labels.

> **Disclaimer:** Personal learning project built with entirely synthetic, programmatically generated data. Not affiliated with, endorsed by, or using systems, schemas, or data from any financial institution.

## How this fits with the rest of the project

| Repo | Stack | Role |
| --- | --- | --- |
| [`aus-banking-pipeline`](https://github.com/vivianasoyoung/aus-banking-pipeline) | Airflow, Postgres, Docker | Foundation: synthetic data generation + batch ingestion |
| [`aus-dbt-analytics`](https://github.com/vivianasoyoung/aus-dbt-analytics) | dbt-postgres, dbt_utils | Staging → intermediate → marts transformations |
| [`aus-fraud-streaming`](https://github.com/vivianasoyoung/aus-fraud-streaming) | Kafka, Python, Postgres | Real-time rule-based fraud detection. Produces the **labels** this repo uses. |
| **[`aus-feature-store`](https://github.com/vivianasoyoung/aus-feature-store)** *(You are here)* | Feast, MLflow, FastAPI | ML feature store + model serving |

> Labels come from `aus-fraud-streaming` (not from the same columns used as features). This decoupling is what makes the model genuinely predictive rather than trivially circular.

---

## Architecture

```
raw transactions (from aus-banking-pipeline)
        +
flagged transactions (labels from aus-fraud-streaming)
        ↓
compute_features.py  ← labels joined from an independent signal
        ↓
Parquet (offline)  →  feast apply / materialize  →  SQLite (online)
        ↓                                              ↓
train_model.py                                     fraud_api.py
   ↓                                                   ↓
MLflow Registry  ──────────────────────────────→  /score endpoint
```

## Tech Stack

| Component | Tool |
|---|---|
| Feature store | Feast (offline Parquet + online SQLite) |
| ML | scikit-learn (RandomForest + LogReg baseline) |
| Experiment tracking | MLflow |
| Serving | FastAPI + Uvicorn |

## Quick Start

```bash
pip install -r requirements.txt

python features/compute_features.py \
    --transactions ../aus-banking-pipeline/data/raw/transactions.csv \
    --flagged ../aus-fraud-streaming/data/flagged_transactions.csv \
    --out feature_repo/data/account_features.parquet

cd feature_repo && feast apply && feast materialize-incremental $(date +%F) && cd ..

mlflow ui --port 5001 &
python training/train_model.py --features feature_repo/data/account_features.parquet

uvicorn serving.fraud_api:app --port 8001
```

## Feature / Label Design

Features describe account behaviour (transaction counts, spend, night/online ratios, etc.). **Labels are sourced from the streaming fraud engine** — an independent signal — not derived from the feature columns. A regression test (`tests/test_compute_features.py`) fails if any single feature can trivially reconstruct the label, guarding against target leakage.

## Expected Metrics

With labels sourced externally, hold-out AUC lands in a realistic ~0.7–0.9 range. (An AUC of 1.00 would indicate leakage — there's a test to catch exactly that.)

## Project Structure

```
aus-feature-store/
├── features/compute_features.py
├── feature_repo/            # Feast definitions + data
├── training/train_model.py
├── serving/fraud_api.py
├── tests/test_compute_features.py
├── requirements.txt
└── README.md
```
