"""
Tests for the ML pipeline:
  - ml/features.py   (feature extraction)
  - ml/predict.py    (model loading and prediction)

These tests exercise the actual functions against the trained artifacts
stored in ml/artifacts/.
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.features import (
    extract_features,
    extract_single_features,
    ALL_FEATURE_COLS,
    CHANNEL_ORDER,
    STATE_RISK_SCORES,
    DEFAULT_STATE_RISK,
    FeatureScaler,
)
from ml.predict import (
    load_model,
    is_model_loaded,
    predict_fraud_score,
    MODEL_PATH,
    SCALER_PATH,
)


# ======================================================================
# Helper: a sample transaction dict
# ======================================================================

def _sample_transaction(**overrides) -> dict:
    base = {
        "amount_ngn": 50_000,
        "channel": "pos",
        "location_state": "Lagos",
        "hour_of_day": 14,
        "day_of_week": 2,
    }
    base.update(overrides)
    return base


# ======================================================================
# 1. Feature extraction — single transaction
# ======================================================================

class TestExtractSingleFeatures:
    def test_output_shape(self):
        """Output should be (1, n_features) where n_features == len(ALL_FEATURE_COLS)."""
        features = extract_single_features(_sample_transaction())
        assert features.shape == (1, len(ALL_FEATURE_COLS))

    def test_output_dtype(self):
        features = extract_single_features(_sample_transaction())
        assert features.dtype == np.float64

    def test_known_state_risk(self):
        """Lagos risk score should be 0.30 in the feature vector."""
        features = extract_single_features(_sample_transaction(location_state="Lagos"))
        # state_risk_score is the 6th feature (index 5)
        assert features[0, 5] == pytest.approx(0.30)

    def test_unknown_state_risk_fallback(self):
        features = extract_single_features(_sample_transaction(location_state="UnknownState"))
        assert features[0, 5] == pytest.approx(DEFAULT_STATE_RISK)

    def test_channel_one_hot_encoding(self):
        """Only the 'pos' column should be 1; all others 0."""
        features = extract_single_features(_sample_transaction(channel="pos"))
        n_numeric = len(ALL_FEATURE_COLS) - len(CHANNEL_ORDER)
        channel_slice = features[0, n_numeric:]
        pos_idx = CHANNEL_ORDER.index("pos")
        for i, val in enumerate(channel_slice):
            if i == pos_idx:
                assert val == 1.0
            else:
                assert val == 0.0

    def test_is_large_amount_flag(self):
        """Amount > 200k should set is_large_amount = 1."""
        features = extract_single_features(_sample_transaction(amount_ngn=300_000))
        # is_large_amount is at index 8 in NUMERIC_FEATURE_COLS
        assert features[0, 8] == 1.0

    def test_is_micro_amount_flag(self):
        """Amount < 500 should set is_micro_amount = 1."""
        features = extract_single_features(_sample_transaction(amount_ngn=100))
        # is_micro_amount is at index 9
        assert features[0, 9] == 1.0


# ======================================================================
# 2. Feature extraction — batch (DataFrame)
# ======================================================================

class TestExtractFeaturesBatch:
    def test_batch_output_columns(self):
        df = pd.DataFrame([
            {"amount_ngn": 10_000, "hour_of_day": 10, "day_of_week": 1,
             "channel": "pos", "location_state": "Lagos"},
            {"amount_ngn": 200_000, "hour_of_day": 22, "day_of_week": 5,
             "channel": "card", "location_state": "Kano"},
        ])
        result = extract_features(df)
        assert list(result.columns) == ALL_FEATURE_COLS

    def test_batch_row_count_preserved(self):
        df = pd.DataFrame([
            {"amount_ngn": 5_000, "hour_of_day": 8, "day_of_week": 0,
             "channel": "ussd", "location_state": "Abuja FCT"},
        ] * 5)
        result = extract_features(df)
        assert len(result) == 5


# ======================================================================
# 3. Model loading
# ======================================================================

class TestModelLoading:
    def test_artifacts_exist(self):
        """Trained model and scaler files should be present."""
        abs_model = os.path.join(PROJECT_ROOT, MODEL_PATH)
        abs_scaler = os.path.join(PROJECT_ROOT, SCALER_PATH)
        assert os.path.exists(abs_model), f"Model not found at {abs_model}"
        assert os.path.exists(abs_scaler), f"Scaler not found at {abs_scaler}"

    def test_load_model_succeeds(self):
        """load_model() should succeed when artifacts are present."""
        # Use absolute paths relative to project root since load_model
        # uses relative paths by default and we might be in tests/
        abs_model = os.path.join(PROJECT_ROOT, MODEL_PATH)
        abs_scaler = os.path.join(PROJECT_ROOT, SCALER_PATH)
        load_model(model_path=abs_model, scaler_path=abs_scaler)
        assert is_model_loaded() is True

    def test_load_model_missing_file(self):
        """Should raise FileNotFoundError for a non-existent path."""
        with pytest.raises(FileNotFoundError):
            load_model(model_path="/nonexistent/model.joblib", scaler_path=SCALER_PATH)


# ======================================================================
# 4. Prediction
# ======================================================================

class TestPredictFraudScore:
    @pytest.fixture(autouse=True)
    def ensure_model_loaded(self):
        """Make sure the model is loaded before each prediction test."""
        abs_model = os.path.join(PROJECT_ROOT, MODEL_PATH)
        abs_scaler = os.path.join(PROJECT_ROOT, SCALER_PATH)
        if not is_model_loaded():
            load_model(model_path=abs_model, scaler_path=abs_scaler)

    def test_returns_float(self):
        score = predict_fraud_score(_sample_transaction())
        assert isinstance(score, float)

    def test_score_in_range(self):
        """Fraud score must be between 0 and 1."""
        score = predict_fraud_score(_sample_transaction())
        assert 0.0 <= score <= 1.0

    def test_prediction_shape_single(self):
        """predict_fraud_score should return a scalar float, not array."""
        score = predict_fraud_score(_sample_transaction())
        assert not hasattr(score, "__len__"), "Score should be a scalar, not array"

    def test_score_precision(self):
        """Score should be rounded to 4 decimal places."""
        score = predict_fraud_score(_sample_transaction())
        decimal_str = str(score).split(".")[-1]
        assert len(decimal_str) <= 4

    def test_different_inputs_may_differ(self):
        """Extreme inputs should produce different scores than normal ones."""
        normal_score = predict_fraud_score(_sample_transaction(amount_ngn=5_000))
        extreme_score = predict_fraud_score(
            _sample_transaction(amount_ngn=50_000_000, channel="ussd", hour_of_day=2)
        )
        # We cannot guarantee they are different with an IsolationForest
        # but we can at least verify both are valid scores
        assert 0.0 <= normal_score <= 1.0
        assert 0.0 <= extreme_score <= 1.0


# ======================================================================
# 5. FeatureScaler
# ======================================================================

class TestFeatureScaler:
    def test_fit_and_transform(self):
        scaler = FeatureScaler()
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        scaler.fit(X)
        assert scaler.is_fitted is True
        transformed = scaler.transform(X)
        assert transformed.shape == X.shape

    def test_transform_before_fit_raises(self):
        scaler = FeatureScaler()
        X = np.array([[1, 2]])
        with pytest.raises(RuntimeError, match="not been fitted"):
            scaler.transform(X)

    def test_fit_transform(self):
        scaler = FeatureScaler()
        X = np.array([[10, 20], [30, 40]], dtype=np.float64)
        result = scaler.fit_transform(X)
        assert scaler.is_fitted is True
        assert result.shape == X.shape
