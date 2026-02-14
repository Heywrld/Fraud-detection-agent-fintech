"""
Synthetic Nigerian fintech transaction data generator.

Generates 10,000+ realistic transactions with fraud patterns
modeled after common Nigerian fintech fraud scenarios.
"""

import os
import uuid
import hashlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
NUM_TRANSACTIONS = 12_000
NUM_CUSTOMERS = 2_000
FRAUD_RATE = 0.05  # ~5 % of transactions are fraudulent

CHANNELS = ["pos", "bank_transfer", "mobile_money", "ussd", "card"]
CHANNEL_WEIGHTS = [0.40, 0.25, 0.20, 0.10, 0.05]

# Amount tier ranges (NGN)
AMOUNT_TIERS = {
    "micro":  (50, 500),
    "small":  (500, 5_000),
    "medium": (5_000, 100_000),
    "large":  (100_000, 1_000_000),
}
AMOUNT_TIER_WEIGHTS = [0.25, 0.35, 0.30, 0.10]

# Nigerian states and their transaction share
STATE_DISTRIBUTION = {
    "Lagos": 0.25,
    "Kano": 0.10,
    "Rivers": 0.08,
    "Abuja FCT": 0.08,
    "Oyo": 0.06,
    # Remaining 32 states share the rest (~43 %)
    "Kaduna": 0.03, "Enugu": 0.025, "Delta": 0.025, "Anambra": 0.025,
    "Edo": 0.02, "Ondo": 0.02, "Osun": 0.02, "Kwara": 0.015,
    "Abia": 0.015, "Imo": 0.015, "Benue": 0.015, "Plateau": 0.015,
    "Bauchi": 0.01, "Borno": 0.01, "Sokoto": 0.01, "Zamfara": 0.01,
    "Kebbi": 0.01, "Niger": 0.01, "Nassarawa": 0.008, "Kogi": 0.008,
    "Taraba": 0.008, "Adamawa": 0.008, "Yobe": 0.007, "Gombe": 0.007,
    "Jigawa": 0.007, "Katsina": 0.007, "Cross River": 0.007,
    "Akwa Ibom": 0.007, "Bayelsa": 0.005, "Ebonyi": 0.005,
    "Ekiti": 0.005, "Ogun": 0.015,
}

# LGAs per state (a small representative sample per state)
STATE_LGAS: dict[str, list[str]] = {
    "Lagos": ["Ikeja", "Eti-Osa", "Surulere", "Lagos Mainland", "Alimosho",
              "Kosofe", "Agege", "Mushin", "Oshodi-Isolo", "Ikorodu"],
    "Kano": ["Nassarawa", "Dala", "Fagge", "Gwale", "Tarauni", "Kumbotso"],
    "Rivers": ["Port-Harcourt", "Obio-Akpor", "Eleme", "Oyigbo"],
    "Abuja FCT": ["Municipal", "Bwari", "Gwagwalada", "Kuje"],
    "Oyo": ["Ibadan North", "Ibadan South-West", "Ogbomoso", "Oyo East"],
    "Kaduna": ["Kaduna North", "Kaduna South", "Zaria"],
    "Enugu": ["Enugu East", "Enugu North", "Nsukka"],
    "Delta": ["Warri South", "Oshimili South", "Uvwie"],
    "Anambra": ["Awka South", "Onitsha North", "Nnewi North"],
    "Edo": ["Oredo", "Egor", "Ikpoba-Okha"],
}

# Distant state pairs for geo-jump fraud
DISTANT_STATE_PAIRS = [
    ("Lagos", "Borno"), ("Lagos", "Sokoto"), ("Rivers", "Kano"),
    ("Enugu", "Zamfara"), ("Abuja FCT", "Bayelsa"), ("Kaduna", "Cross River"),
    ("Kano", "Delta"), ("Oyo", "Adamawa"), ("Lagos", "Yobe"),
]


def _generate_device_fingerprint(customer_id: str, device_idx: int = 0) -> str:
    """Deterministic device fingerprint per customer."""
    raw = f"{customer_id}:device:{device_idx}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _pick_lga(state: str, rng: np.random.RandomState) -> str:
    """Pick a random LGA for the given state."""
    lgas = STATE_LGAS.get(state)
    if lgas:
        return rng.choice(lgas)
    return f"{state}-Central"


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def generate_dataset(
    n_transactions: int = NUM_TRANSACTIONS,
    n_customers: int = NUM_CUSTOMERS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate a synthetic Nigerian fintech transaction dataset.

    Returns a DataFrame with columns:
        transaction_id, customer_id, amount_ngn, channel, location_state,
        location_lga, device_fingerprint, timestamp, hour_of_day,
        day_of_week, is_fraud
    """
    rng = np.random.RandomState(seed)

    # Pre-assign each customer a "home" state, preferred channel, and normal
    # daily transaction count so we can model behavioural anomalies later.
    customer_ids = [f"CUS-{str(i).zfill(5)}" for i in range(n_customers)]
    states_list = list(STATE_DISTRIBUTION.keys())
    state_probs = np.array(list(STATE_DISTRIBUTION.values()))
    state_probs = state_probs / state_probs.sum()  # normalise

    customer_home_state = {
        cid: rng.choice(states_list, p=state_probs) for cid in customer_ids
    }
    customer_preferred_channel = {
        cid: rng.choice(CHANNELS, p=CHANNEL_WEIGHTS) for cid in customer_ids
    }
    customer_normal_tx_per_day = {
        cid: rng.choice([2, 3]) for cid in customer_ids
    }
    customer_device = {
        cid: _generate_device_fingerprint(cid) for cid in customer_ids
    }

    # ---- Generate legitimate transactions first ----
    n_legit = int(n_transactions * (1 - FRAUD_RATE))
    n_fraud = n_transactions - n_legit

    base_date = datetime(2025, 1, 1)
    span_days = 90  # 3 months of data

    records: list[dict] = []

    for _ in range(n_legit):
        cid = rng.choice(customer_ids)
        state = customer_home_state[cid]
        channel = customer_preferred_channel[cid]

        # Occasionally use a different channel (20 % of the time)
        if rng.random() < 0.20:
            channel = rng.choice(CHANNELS, p=CHANNEL_WEIGHTS)

        # Pick amount tier
        tier_name = rng.choice(list(AMOUNT_TIERS.keys()), p=AMOUNT_TIER_WEIGHTS)
        lo, hi = AMOUNT_TIERS[tier_name]
        amount = round(rng.uniform(lo, hi), 2)

        # Timestamp: bias toward 9am-9pm WAT
        day_offset = rng.randint(0, span_days)
        if rng.random() < 0.80:
            hour = rng.choice(range(9, 22))  # peak hours
        else:
            hour = rng.randint(0, 24)
        minute = rng.randint(0, 60)
        second = rng.randint(0, 60)
        ts = base_date + timedelta(days=int(day_offset), hours=int(hour),
                                   minutes=int(minute), seconds=int(second))

        records.append({
            "transaction_id": str(uuid.uuid4()),
            "customer_id": cid,
            "amount_ngn": amount,
            "channel": channel,
            "location_state": state,
            "location_lga": _pick_lga(state, rng),
            "device_fingerprint": customer_device[cid],
            "timestamp": ts,
            "hour_of_day": ts.hour,
            "day_of_week": ts.weekday(),
            "is_fraud": False,
        })

    # ---- Generate fraud transactions ----
    fraud_per_pattern = n_fraud // 5
    remainder = n_fraud - fraud_per_pattern * 5

    # Pattern 1: Rapid micro-transactions (5+ txns under 500 NGN within 1 hr)
    for i in range(fraud_per_pattern + (1 if remainder > 0 else 0)):
        cid = rng.choice(customer_ids)
        state = customer_home_state[cid]
        day_offset = rng.randint(0, span_days)
        base_hour = rng.randint(0, 23)
        base_ts = base_date + timedelta(days=int(day_offset), hours=int(base_hour))

        burst_count = rng.randint(5, 9)
        for j in range(burst_count):
            ts = base_ts + timedelta(minutes=int(rng.randint(0, 55)))
            amount = round(rng.uniform(50, 499), 2)
            records.append({
                "transaction_id": str(uuid.uuid4()),
                "customer_id": cid,
                "amount_ngn": amount,
                "channel": rng.choice(["pos", "ussd", "mobile_money"]),
                "location_state": state,
                "location_lga": _pick_lga(state, rng),
                "device_fingerprint": customer_device[cid],
                "timestamp": ts,
                "hour_of_day": ts.hour,
                "day_of_week": ts.weekday(),
                "is_fraud": True,
            })
    remainder = max(0, remainder - 1)

    # Pattern 2: Geo-jump (transactions from 2 distant states within 30 min)
    for i in range(fraud_per_pattern + (1 if remainder > 0 else 0)):
        cid = rng.choice(customer_ids)
        pair = DISTANT_STATE_PAIRS[rng.randint(0, len(DISTANT_STATE_PAIRS))]
        day_offset = rng.randint(0, span_days)
        base_hour = rng.randint(6, 22)
        base_ts = base_date + timedelta(days=int(day_offset), hours=int(base_hour))

        for idx, state in enumerate(pair):
            ts = base_ts + timedelta(minutes=int(idx * rng.randint(5, 25)))
            tier_name = rng.choice(["small", "medium"])
            lo, hi = AMOUNT_TIERS[tier_name]
            amount = round(rng.uniform(lo, hi), 2)
            records.append({
                "transaction_id": str(uuid.uuid4()),
                "customer_id": cid,
                "amount_ngn": amount,
                "channel": rng.choice(CHANNELS, p=CHANNEL_WEIGHTS),
                "location_state": state,
                "location_lga": _pick_lga(state, rng),
                "device_fingerprint": customer_device[cid],
                "timestamp": ts,
                "hour_of_day": ts.hour,
                "day_of_week": ts.weekday(),
                "is_fraud": True,
            })
    remainder = max(0, remainder - 1)

    # Pattern 3: Odd-hour large transfers (>200K NGN between midnight and 5am)
    for i in range(fraud_per_pattern + (1 if remainder > 0 else 0)):
        cid = rng.choice(customer_ids)
        state = customer_home_state[cid]
        day_offset = rng.randint(0, span_days)
        hour = rng.randint(0, 5)
        minute = rng.randint(0, 59)
        ts = base_date + timedelta(days=int(day_offset), hours=int(hour),
                                   minutes=int(minute))
        amount = round(rng.uniform(200_001, 950_000), 2)
        records.append({
            "transaction_id": str(uuid.uuid4()),
            "customer_id": cid,
            "amount_ngn": amount,
            "channel": "bank_transfer",
            "location_state": state,
            "location_lga": _pick_lga(state, rng),
            "device_fingerprint": customer_device[cid],
            "timestamp": ts,
            "hour_of_day": ts.hour,
            "day_of_week": ts.weekday(),
            "is_fraud": True,
        })
    remainder = max(0, remainder - 1)

    # Pattern 4: Velocity spike (customer goes from 2-3 tx/day to 10+ in a day)
    for i in range(fraud_per_pattern + (1 if remainder > 0 else 0)):
        cid = rng.choice(customer_ids)
        state = customer_home_state[cid]
        day_offset = rng.randint(0, span_days)
        spike_count = rng.randint(10, 16)
        for j in range(spike_count):
            hour = rng.randint(6, 23)
            minute = rng.randint(0, 59)
            ts = base_date + timedelta(days=int(day_offset), hours=int(hour),
                                       minutes=int(minute))
            tier_name = rng.choice(list(AMOUNT_TIERS.keys()), p=AMOUNT_TIER_WEIGHTS)
            lo, hi = AMOUNT_TIERS[tier_name]
            amount = round(rng.uniform(lo, hi), 2)
            records.append({
                "transaction_id": str(uuid.uuid4()),
                "customer_id": cid,
                "amount_ngn": amount,
                "channel": rng.choice(CHANNELS, p=CHANNEL_WEIGHTS),
                "location_state": state,
                "location_lga": _pick_lga(state, rng),
                "device_fingerprint": customer_device[cid],
                "timestamp": ts,
                "hour_of_day": ts.hour,
                "day_of_week": ts.weekday(),
                "is_fraud": True,
            })
    remainder = max(0, remainder - 1)

    # Pattern 5: Channel anomaly (POS customer suddenly does large bank_transfer)
    for i in range(fraud_per_pattern + (1 if remainder > 0 else 0)):
        # Pick a customer whose preferred channel is POS
        pos_customers = [c for c in customer_ids
                         if customer_preferred_channel[c] == "pos"]
        if not pos_customers:
            pos_customers = customer_ids
        cid = rng.choice(pos_customers)
        state = customer_home_state[cid]
        day_offset = rng.randint(0, span_days)
        hour = rng.randint(0, 23)
        minute = rng.randint(0, 59)
        ts = base_date + timedelta(days=int(day_offset), hours=int(hour),
                                   minutes=int(minute))
        amount = round(rng.uniform(150_000, 900_000), 2)
        records.append({
            "transaction_id": str(uuid.uuid4()),
            "customer_id": cid,
            "amount_ngn": amount,
            "channel": "bank_transfer",
            "location_state": state,
            "location_lga": _pick_lga(state, rng),
            "device_fingerprint": customer_device[cid],
            "timestamp": ts,
            "hour_of_day": ts.hour,
            "day_of_week": ts.weekday(),
            "is_fraud": True,
        })

    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_dataset(df: pd.DataFrame | None = None, path: str = "ml/artifacts/synthetic_data.csv") -> str:
    """
    Save the dataset to CSV.  If no DataFrame is supplied, one is generated.
    Returns the absolute path of the saved file.
    """
    if df is None:
        df = generate_dataset()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    abs_path = os.path.abspath(path)
    print(f"[synthetic_data] Saved {len(df):,} transactions to {abs_path}")
    return abs_path


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[synthetic_data] Generating dataset ...")
    df = generate_dataset()
    print(f"[synthetic_data] Total transactions : {len(df):,}")
    print(f"[synthetic_data] Fraud transactions : {df['is_fraud'].sum():,} "
          f"({df['is_fraud'].mean():.1%})")
    print(f"[synthetic_data] Channels           : {df['channel'].value_counts().to_dict()}")
    print(f"[synthetic_data] Top states          : {df['location_state'].value_counts().head(5).to_dict()}")
    save_dataset(df)
