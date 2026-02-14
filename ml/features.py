"""
Feature engineering pipeline for the Fraud Guardian fraud detection model.

Provides both batch (DataFrame) and single-transaction feature extraction,
plus a StandardScaler wrapper for consistent scaling.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Pre-computed risk scores per Nigerian state
# Higher values => historically higher fraud incidence in synthetic data.
# ---------------------------------------------------------------------------

STATE_RISK_SCORES: dict[str, float] = {
    "Lagos": 0.30,
    "Rivers": 0.25,
    "Abuja FCT": 0.22,
    "Kano": 0.20,
    "Delta": 0.20,
    "Oyo": 0.18,
    "Anambra": 0.18,
    "Enugu": 0.16,
    "Kaduna": 0.15,
    "Edo": 0.15,
    "Ondo": 0.14,
    "Osun": 0.13,
    "Kwara": 0.12,
    "Ogun": 0.12,
    "Abia": 0.12,
    "Imo": 0.11,
    "Benue": 0.10,
    "Plateau": 0.10,
    "Cross River": 0.10,
    "Akwa Ibom": 0.10,
    "Bauchi": 0.09,
    "Borno": 0.09,
    "Sokoto": 0.08,
    "Zamfara": 0.08,
    "Kebbi": 0.08,
    "Niger": 0.08,
    "Nassarawa": 0.08,
    "Kogi": 0.08,
    "Taraba": 0.07,
    "Adamawa": 0.07,
    "Yobe": 0.07,
    "Gombe": 0.07,
    "Jigawa": 0.07,
    "Katsina": 0.07,
    "Bayelsa": 0.06,
    "Ebonyi": 0.06,
    "Ekiti": 0.06,
}

DEFAULT_STATE_RISK = 0.10  # fallback for unknown states

# Canonical channel list for one-hot encoding (alphabetical order)
CHANNEL_ORDER = ["bank_transfer", "card", "mobile_money", "pos", "ussd"]

# ---------------------------------------------------------------------------
# Feature column names produced by this pipeline
# ---------------------------------------------------------------------------

NUMERIC_FEATURE_COLS = [
    "amount_log",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "state_risk_score",
    "amount_zscore",
    "is_odd_hour",
    "is_large_amount",
    "is_micro_amount",
]

CHANNEL_FEATURE_COLS = [f"channel_{c}" for c in CHANNEL_ORDER]

ALL_FEATURE_COLS = NUMERIC_FEATURE_COLS + CHANNEL_FEATURE_COLS


# ---------------------------------------------------------------------------
# Batch feature extraction
# ---------------------------------------------------------------------------

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform a raw transaction DataFrame into a feature matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at minimum: amount_ngn, hour_of_day, day_of_week,
        channel, location_state.

    Returns
    -------
    pd.DataFrame with columns listed in ``ALL_FEATURE_COLS``.
    """
    features = pd.DataFrame(index=df.index)

    # --- Continuous features ---
    features["amount_log"] = np.log1p(df["amount_ngn"].astype(float))

    # Cyclical encoding of hour (period = 24)
    hour = df["hour_of_day"].astype(float)
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    # Cyclical encoding of day of week (period = 7)
    day = df["day_of_week"].astype(float)
    features["day_sin"] = np.sin(2 * np.pi * day / 7.0)
    features["day_cos"] = np.cos(2 * np.pi * day / 7.0)

    # State risk score
    features["state_risk_score"] = (
        df["location_state"]
        .map(STATE_RISK_SCORES)
        .fillna(DEFAULT_STATE_RISK)
    )

    # Amount z-score *within the same channel*
    channel_groups = df.groupby("channel")["amount_ngn"]
    channel_mean = channel_groups.transform("mean")
    channel_std = channel_groups.transform("std").replace(0, 1)  # avoid div/0
    features["amount_zscore"] = (df["amount_ngn"] - channel_mean) / channel_std

    # Binary indicators
    features["is_odd_hour"] = (df["hour_of_day"].astype(int).between(0, 5)).astype(int)
    features["is_large_amount"] = (df["amount_ngn"].astype(float) > 200_000).astype(int)
    features["is_micro_amount"] = (df["amount_ngn"].astype(float) < 500).astype(int)

    # --- One-hot encoding of channel ---
    for ch in CHANNEL_ORDER:
        features[f"channel_{ch}"] = (df["channel"] == ch).astype(int)

    # Guarantee column order
    features = features[ALL_FEATURE_COLS]
    return features


# ---------------------------------------------------------------------------
# Single-transaction feature extraction (for real-time prediction)
# ---------------------------------------------------------------------------

def extract_single_features(transaction: dict) -> np.ndarray:
    """
    Extract features from a single transaction dictionary.

    Parameters
    ----------
    transaction : dict
        Must contain keys: amount_ngn, hour_of_day, day_of_week,
        channel, location_state.
        ``hour_of_day`` and ``day_of_week`` will be derived from
        ``timestamp`` if not explicitly provided.

    Returns
    -------
    np.ndarray of shape (1, n_features) -- a single-row feature vector.
    """
    amount = float(transaction["amount_ngn"])
    hour = int(transaction.get("hour_of_day", 12))
    day = int(transaction.get("day_of_week", 0))
    channel = str(transaction.get("channel", "pos"))
    state = str(transaction.get("location_state", "Lagos"))

    vec: list[float] = []

    # amount_log
    vec.append(float(np.log1p(amount)))

    # hour cyclical
    vec.append(float(np.sin(2 * np.pi * hour / 24.0)))
    vec.append(float(np.cos(2 * np.pi * hour / 24.0)))

    # day cyclical
    vec.append(float(np.sin(2 * np.pi * day / 7.0)))
    vec.append(float(np.cos(2 * np.pi * day / 7.0)))

    # state risk
    vec.append(STATE_RISK_SCORES.get(state, DEFAULT_STATE_RISK))

    # amount_zscore -- for a single transaction without group context we
    # use a naive z-score against a channel-level prior.
    _CHANNEL_AMOUNT_MEAN = {
        "pos": 12_000, "bank_transfer": 85_000, "mobile_money": 8_000,
        "ussd": 3_000, "card": 25_000,
    }
    _CHANNEL_AMOUNT_STD = {
        "pos": 20_000, "bank_transfer": 120_000, "mobile_money": 15_000,
        "ussd": 5_000, "card": 40_000,
    }
    ch_mean = _CHANNEL_AMOUNT_MEAN.get(channel, 20_000)
    ch_std = _CHANNEL_AMOUNT_STD.get(channel, 30_000)
    vec.append((amount - ch_mean) / ch_std)

    # binary indicators
    vec.append(1.0 if 0 <= hour <= 5 else 0.0)
    vec.append(1.0 if amount > 200_000 else 0.0)
    vec.append(1.0 if amount < 500 else 0.0)

    # channel one-hot
    for ch in CHANNEL_ORDER:
        vec.append(1.0 if channel == ch else 0.0)

    return np.array(vec, dtype=np.float64).reshape(1, -1)


# ---------------------------------------------------------------------------
# Scaler wrapper
# ---------------------------------------------------------------------------

class FeatureScaler:
    """Thin wrapper around sklearn StandardScaler with project-specific helpers."""

    def __init__(self):
        self.scaler = StandardScaler()
        self._is_fitted = False

    def fit(self, X: pd.DataFrame | np.ndarray) -> "FeatureScaler":
        self.scaler.fit(X)
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("FeatureScaler has not been fitted yet. Call fit() first.")
        return self.scaler.transform(X)

    def fit_transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
