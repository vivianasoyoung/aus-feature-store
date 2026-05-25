"""
train_model.py
--------------
Trains a fraud detection classifier.

Labels are EXTERNAL to the features (sourced from the streaming fraud
engine via cba-fraud-streaming → compute_features.py). This decouples
label definition from feature definition and is what makes the task
genuinely predictive instead of trivially circular.

Logs experiment to MLflow and registers the model under `cba_fraud_model`.
"""

import argparse

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REGISTERED_MODEL_NAME = "cba_fraud_model"
EXPERIMENT_NAME = "cba_fraud_detection"

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
LABEL_COL = "is_fraud_account"


def load(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if LABEL_COL not in df.columns:
        raise ValueError(
            f"Expected label column '{LABEL_COL}' in {path}. "
            f"Did you run compute_features.py with --flagged?"
        )
    return df


def train(features_path: str) -> str:
    df = load(features_path)
    X = df[FEATURE_COLS]
    y = df[LABEL_COL]

    pos_rate = y.mean()
    print(f"Loaded {len(df):,} rows; positive label rate = {pos_rate:.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run() as run:
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_param("positive_rate", round(pos_rate, 4))

        # ── Baseline ──────────────────────────────────────────────────────────
        baseline = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ])
        cv_scores = cross_val_score(
            baseline, X_train, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring="roc_auc",
        )
        mlflow.log_metric("baseline_cv_auc_mean", cv_scores.mean())
        mlflow.log_metric("baseline_cv_auc_std", cv_scores.std())
        print(f"Baseline (LogReg) CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # ── Candidate model ───────────────────────────────────────────────────
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        rf_cv = cross_val_score(
            rf, X_train, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring="roc_auc",
        )
        mlflow.log_metric("rf_cv_auc_mean", rf_cv.mean())
        mlflow.log_metric("rf_cv_auc_std", rf_cv.std())
        print(f"RandomForest CV AUC: {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")

        # Final fit on full train set, evaluate on held-out test
        rf.fit(X_train, y_train)
        test_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
        mlflow.log_metric("test_auc", test_auc)
        print(f"Hold-out test AUC: {test_auc:.4f}")
        print(classification_report(y_test, rf.predict(X_test)))

        for feat, imp in sorted(zip(FEATURE_COLS, rf.feature_importances_), key=lambda x: -x[1]):
            mlflow.log_metric(f"importance_{feat}", round(float(imp), 4))

        # Register the model
        mlflow.sklearn.log_model(
            rf,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        client = MlflowClient()
        latest = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["None"])[0]
        client.transition_model_version_stage(
            REGISTERED_MODEL_NAME, latest.version, stage="Staging",
            archive_existing_versions=True,
        )
        print(f"Registered {REGISTERED_MODEL_NAME} v{latest.version} → Staging")

        return run.info.run_id


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True, help="Path to features parquet")
    args = p.parse_args()
    train(args.features)


if __name__ == "__main__":
    main()
