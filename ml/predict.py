"""
Real-time fraud prediction pipeline.

Loads the trained IsolationForest model and scaler once, then exposes a
fast ``predict_fraud_score()`` function for individual transactions.
"""

import os

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ml.features import extract_single_features

# ---------------------------------------------------------------------------
# Global model / scaler singletons (loaded once via load_model())
# ---------------------------------------------------------------------------

_model: IsolationForest | None = None
_scaler: StandardScaler | None = None

ARTIFACTS_DIR = os.path.join("ml", "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "fraud_model.joblib")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.joblib")


def load_model(
    model_path: str = MODEL_PATH,
    scaler_path: str = SCALER_PATH,
) -> None:
    """
    Load the trained model and scaler from disk into global variables.

    This should be called once at application startup (e.g. in the
    FastAPI lifespan handler).

    Raises
    ------
    FileNotFoundError
        If either the model or scaler artifact is missing.
    """
    global _model, _scaler

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model artifact not found at '{os.path.abspath(model_path)}'. "
            "Train the model first with:  python -m ml.train"
        )
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler artifact not found at '{os.path.abspath(scaler_path)}'. "
            "Train the model first with:  python -m ml.train"
        )

    _model = joblib.load(model_path)
    _scaler = joblib.load(scaler_path)


def is_model_loaded() -> bool:
    """Return True if both model and scaler are loaded."""
    return _model is not None and _scaler is not None


def predict_fraud_score(transaction: dict) -> float:
    """
    Predict a fraud score for a single transaction.

    Parameters
    ----------
    transaction : dict
        Must contain at minimum: amount_ngn, channel, location_state.
        Optionally: hour_of_day, day_of_week (defaults applied if absent).

    Returns
    -------
    float
        A score in the range [0.0, 1.0] where:
          - 0.0 = very likely legitimate
          - 1.0 = very likely fraudulent

    Raises
    ------
    RuntimeError
        If the model has not been loaded yet.
    """
    if _model is None or _scaler is None:
        raise RuntimeError(
            "Fraud model is not loaded. Call load_model() first, or run "
            "'python -m ml.train' to create the model artifacts."
        )

    # 1. Extract feature vector (shape: 1 x n_features)
    raw_features = extract_single_features(transaction)

    # 2. Scale using the fitted scaler
    scaled_features = _scaler.transform(raw_features)

    # 3. Get the anomaly decision function value.
    #    IsolationForest.decision_function returns negative values for
    #    anomalies and positive values for normal samples.  More negative
    #    means more anomalous.
    decision_value = _model.decision_function(scaled_features)[0]

    # 4. Convert to a [0, 1] fraud probability.
    #    Strategy: use a sigmoid-like mapping centred on 0.
    #    decision_function ≈ 0  -> score ≈ 0.5  (borderline)
    #    decision_function << 0 -> score -> 1.0  (fraud)
    #    decision_function >> 0 -> score -> 0.0  (legit)
    #
    #    We use:  score = 1 / (1 + exp(k * decision_value))
    #    with k = 5 to sharpen the transition.
    k = 5.0
    score = float(1.0 / (1.0 + np.exp(k * decision_value)))

    # Clamp to [0, 1] for safety
    score = max(0.0, min(1.0, score))
    return round(score, 4)
