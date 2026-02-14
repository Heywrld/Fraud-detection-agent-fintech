"""
Train the Isolation Forest fraud detection model.

Usage
-----
    python -m ml.train

Outputs (all written to ml/artifacts/):
    fraud_model.joblib   -- trained IsolationForest
    scaler.joblib        -- fitted StandardScaler
    model_metrics.json   -- precision, recall, F1, bias report
    synthetic_data.csv   -- generated training data (if not already present)
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from ml.features import ALL_FEATURE_COLS, FeatureScaler, extract_features
from ml.synthetic_data import generate_dataset, save_dataset

ARTIFACTS_DIR = os.path.join("ml", "artifacts")
DATA_PATH = os.path.join(ARTIFACTS_DIR, "synthetic_data.csv")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "fraud_model.joblib")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.joblib")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "model_metrics.json")


def _load_or_generate_data() -> pd.DataFrame:
    """Load synthetic data from disk, or generate + save it."""
    if os.path.exists(DATA_PATH):
        print(f"[train] Loading existing data from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        print("[train] No data file found -- generating synthetic dataset ...")
        df = generate_dataset()
        save_dataset(df, DATA_PATH)
    return df


def _evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Compute precision, recall, F1 for the fraud class."""
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }


def _bias_report(df: pd.DataFrame, y_pred: np.ndarray) -> dict:
    """
    Check whether the model flags certain states or channels
    disproportionately compared to the base fraud rate.
    """
    df_eval = df.copy()
    df_eval["predicted_fraud"] = y_pred

    base_flag_rate = float(y_pred.mean())

    # --- State bias ---
    state_rates = (
        df_eval.groupby("location_state")["predicted_fraud"]
        .mean()
        .sort_values(ascending=False)
    )
    state_bias: dict[str, float] = {}
    for state, rate in state_rates.items():
        ratio = rate / base_flag_rate if base_flag_rate > 0 else 0.0
        if ratio > 1.5 or ratio < 0.5:
            state_bias[state] = round(float(rate), 4)

    # --- Channel bias ---
    channel_rates = (
        df_eval.groupby("channel")["predicted_fraud"]
        .mean()
        .sort_values(ascending=False)
    )
    channel_bias: dict[str, float] = {}
    for channel, rate in channel_rates.items():
        ratio = rate / base_flag_rate if base_flag_rate > 0 else 0.0
        if ratio > 1.5 or ratio < 0.5:
            channel_bias[channel] = round(float(rate), 4)

    return {
        "base_flag_rate": round(base_flag_rate, 4),
        "states_disproportionately_flagged": state_bias,
        "channels_disproportionately_flagged": channel_bias,
    }


def train() -> dict:
    """
    Full training pipeline.  Returns the metrics dictionary.
    """
    print("=" * 60)
    print("  Fraud Guardian AI -- Model Training")
    print("=" * 60)

    # 1. Data -----------------------------------------------------------
    df = _load_or_generate_data()
    print(f"[train] Dataset size  : {len(df):,} transactions")
    print(f"[train] Fraud count   : {df['is_fraud'].sum():,} "
          f"({df['is_fraud'].mean():.2%})")

    # 2. Features -------------------------------------------------------
    print("[train] Extracting features ...")
    X = extract_features(df)
    y = df["is_fraud"].astype(int).values

    # 3. Scale ----------------------------------------------------------
    print("[train] Fitting scaler ...")
    scaler = FeatureScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Train ----------------------------------------------------------
    print("[train] Training IsolationForest (n_estimators=200, contamination=0.05) ...")
    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # 5. Evaluate -------------------------------------------------------
    # IsolationForest returns -1 for anomalies, 1 for normal
    raw_preds = model.predict(X_scaled)
    y_pred = (raw_preds == -1).astype(int)  # 1 = fraud

    metrics = _evaluate(y, y_pred)
    print(f"[train] Precision : {metrics['precision']:.4f}")
    print(f"[train] Recall    : {metrics['recall']:.4f}")
    print(f"[train] F1 Score  : {metrics['f1_score']:.4f}")

    print("\n[train] Classification report:")
    print(classification_report(y, y_pred, target_names=["legit", "fraud"]))

    # 6. Bias report ----------------------------------------------------
    print("[train] Generating bias report ...")
    bias = _bias_report(df, y_pred)
    metrics["bias_report"] = bias

    if bias["states_disproportionately_flagged"]:
        print(f"[train] WARNING -- States disproportionately flagged: "
              f"{list(bias['states_disproportionately_flagged'].keys())}")
    else:
        print("[train] No significant state-level bias detected.")

    if bias["channels_disproportionately_flagged"]:
        print(f"[train] WARNING -- Channels disproportionately flagged: "
              f"{list(bias['channels_disproportionately_flagged'].keys())}")
    else:
        print("[train] No significant channel-level bias detected.")

    # 7. Persist --------------------------------------------------------
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    print(f"[train] Model saved  -> {os.path.abspath(MODEL_PATH)}")

    joblib.dump(scaler.scaler, SCALER_PATH)
    print(f"[train] Scaler saved -> {os.path.abspath(SCALER_PATH)}")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train] Metrics saved -> {os.path.abspath(METRICS_PATH)}")

    print("=" * 60)
    print("  Training complete!")
    print("=" * 60)
    return metrics


# ---------------------------------------------------------------------------
# CLI entry-point:  python -m ml.train
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
