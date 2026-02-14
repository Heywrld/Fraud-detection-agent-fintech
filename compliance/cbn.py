"""
CBN (Central Bank of Nigeria) regulatory compliance utilities.

Provides transaction risk categorisation and Suspicious Transaction Report
(STR) flag logic aligned with:

- CBN AML/CFT Regulations 2013 (amended 2022)
- CBN Risk-Based Approach to AML/CFT (Guidance Note)
- Money Laundering (Prohibition) Act 2011 (as amended 2022)
- NFIU Act 2018 -- STR filing requirements
- CBN Circular on Cash-Based Transaction Reporting

All functions are *pure* (no DB or network calls) so they remain fully
testable by the services layer.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from compliance.ndpr import CBN_SINGLE_TRANSACTION_LIMIT, CBN_DAILY_CUMULATIVE_LIMIT


# ============================================================================
# 1. Risk-Level Categorisation
# ============================================================================

class RiskLevel(str, Enum):
    """CBN risk classification tiers for transactions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Channels ordered roughly by risk profile.
# POS and bank transfers have stronger identity assurance; USSD and mobile
# money are easier to use with stolen SIMs / compromised accounts.
_HIGH_RISK_CHANNELS = {"ussd", "mobile_money"}
_MEDIUM_RISK_CHANNELS = {"card"}
_LOW_RISK_CHANNELS = {"pos", "bank_transfer"}

# CBN guidance treats transactions outside normal business hours (09:00-17:00
# WAT) as elevated-risk.  Transactions between midnight and 05:00 are
# particularly suspicious.
_ODD_HOUR_START = 0   # midnight
_ODD_HOUR_END = 5     # 05:00


def categorize_transaction_risk(
    amount: float,
    channel: str,
    hour_of_day: int,
    daily_total: float = 0.0,
    fraud_score: float = 0.0,
) -> dict:
    """Assign a CBN risk level to a transaction.

    The risk score is additive -- each risk factor contributes points
    and the total maps to a tier:

    * 0-2  -> LOW
    * 3-4  -> MEDIUM
    * 5-7  -> HIGH
    * 8+   -> CRITICAL

    Parameters
    ----------
    amount : float
        Transaction amount in Naira.
    channel : str
        Payment channel (pos, card, ussd, mobile_money, bank_transfer).
    hour_of_day : int
        Hour of day in WAT (0-23).
    daily_total : float, optional
        Cumulative daily amount for this customer (including this txn).
    fraud_score : float, optional
        ML fraud score (0.0 -- 1.0) if already computed.

    Returns
    -------
    dict
        ``risk_level`` : RiskLevel enum value
        ``risk_score`` : int (raw additive score)
        ``factors``    : list[str] -- human-readable risk factors
    """
    score = 0
    factors: list[str] = []

    # -- Amount-based risk ------------------------------------------------
    if amount >= CBN_SINGLE_TRANSACTION_LIMIT:
        score += 3
        factors.append(
            f"Amount N{amount:,.2f} meets/exceeds CBN N5M EDD threshold"
        )
    elif amount >= 1_000_000:
        score += 2
        factors.append(f"Large transaction: N{amount:,.2f}")
    elif amount >= 500_000:
        score += 1
        factors.append(f"Moderately large transaction: N{amount:,.2f}")

    # -- Cumulative daily risk --------------------------------------------
    if daily_total >= CBN_DAILY_CUMULATIVE_LIMIT:
        score += 3
        factors.append(
            f"Daily cumulative N{daily_total:,.2f} meets/exceeds CBN N10M CTR threshold"
        )
    elif daily_total >= CBN_SINGLE_TRANSACTION_LIMIT:
        score += 1
        factors.append(f"Daily cumulative N{daily_total:,.2f} approaching CTR threshold")

    # -- Channel risk -----------------------------------------------------
    channel_lower = channel.lower() if channel else ""
    if channel_lower in _HIGH_RISK_CHANNELS:
        score += 2
        factors.append(f"High-risk channel: {channel}")
    elif channel_lower in _MEDIUM_RISK_CHANNELS:
        score += 1
        factors.append(f"Medium-risk channel: {channel}")
    # Low-risk channels contribute 0

    # -- Time-of-day risk -------------------------------------------------
    if _ODD_HOUR_START <= hour_of_day < _ODD_HOUR_END:
        score += 2
        factors.append(f"Unusual hour: {hour_of_day:02d}:00 WAT (midnight-05:00)")
    elif hour_of_day < 6 or hour_of_day >= 23:
        score += 1
        factors.append(f"Outside normal business hours: {hour_of_day:02d}:00 WAT")

    # -- ML fraud score boost ---------------------------------------------
    if fraud_score >= 0.9:
        score += 3
        factors.append(f"ML fraud score critical: {fraud_score:.2f}")
    elif fraud_score >= 0.7:
        score += 2
        factors.append(f"ML fraud score elevated: {fraud_score:.2f}")
    elif fraud_score >= 0.5:
        score += 1
        factors.append(f"ML fraud score moderate: {fraud_score:.2f}")

    # -- Map raw score to tier --------------------------------------------
    if score >= 8:
        level = RiskLevel.CRITICAL
    elif score >= 5:
        level = RiskLevel.HIGH
    elif score >= 3:
        level = RiskLevel.MEDIUM
    else:
        level = RiskLevel.LOW

    if not factors:
        factors.append("Transaction within normal parameters")

    return {
        "risk_level": level,
        "risk_score": score,
        "factors": factors,
    }


# ============================================================================
# 2. Suspicious Transaction Report (STR) Flag Logic
# ============================================================================
#
# Under Section 6 of the Money Laundering (Prohibition) Act 2011 and
# NFIU Act 2018, financial institutions must file an STR with the NFIU
# when a transaction is "suspected to involve proceeds of an unlawful
# act or is intended to be used in furtherance of terrorism."
#
# The CBN AML/CFT Regulation, Section 3, enumerates red-flag indicators.
# We codify the key objective ones below.
# ============================================================================

# Red-flag: rapid-fire transactions within a short window may indicate
# structuring (smurfing) to stay below reporting thresholds.
STRUCTURING_COUNT_THRESHOLD = 5          # 5+ transactions in a day
STRUCTURING_AMOUNT_PROXIMITY = 0.80      # each txn within 80% of the N5M limit

# Red-flag: geographic mismatch -- if a customer transacts from a state
# that differs from their registered state, it is a potential indicator.
# (In production this would be driven by a customer profile lookup.)


def evaluate_str_requirement(
    amount: float,
    channel: str,
    hour_of_day: int,
    fraud_score: float = 0.0,
    daily_total: float = 0.0,
    daily_transaction_count: int = 1,
    customer_state: Optional[str] = None,
    transaction_state: Optional[str] = None,
) -> dict:
    """Determine whether a Suspicious Transaction Report must be filed.

    This function evaluates objective red-flag indicators from CBN
    AML/CFT guidance and returns an advisory result.  The final
    decision to file an STR rests with the Compliance Officer, but
    this function automates the initial screening.

    Parameters
    ----------
    amount : float
        Transaction amount in Naira.
    channel : str
        Payment channel identifier.
    hour_of_day : int
        Hour of the transaction in WAT (0-23).
    fraud_score : float
        ML-computed fraud probability (0.0-1.0).
    daily_total : float
        Customer's cumulative daily transaction total (incl. this txn).
    daily_transaction_count : int
        Number of transactions by this customer today (incl. this one).
    customer_state : str, optional
        Customer's registered state of residence.
    transaction_state : str, optional
        State where this transaction originated.

    Returns
    -------
    dict
        ``file_str``       -- bool, True if an STR filing is recommended.
        ``risk_level``     -- RiskLevel enum value from categorisation.
        ``red_flags``      -- list[str] of triggered indicators.
        ``recommendation`` -- str, human-readable next-step guidance.
    """
    red_flags: list[str] = []

    # -- 1. High ML fraud score -------------------------------------------
    if fraud_score >= 0.9:
        red_flags.append(
            f"Critical ML fraud score ({fraud_score:.2f}) -- possible "
            "fraudulent activity"
        )
    elif fraud_score >= 0.7:
        red_flags.append(
            f"Elevated ML fraud score ({fraud_score:.2f}) -- warrants "
            "manual review"
        )

    # -- 2. Amount exceeds CBN single-transaction limit -------------------
    if amount >= CBN_SINGLE_TRANSACTION_LIMIT:
        red_flags.append(
            f"Single transaction N{amount:,.2f} meets/exceeds N5M CBN "
            "reporting threshold"
        )

    # -- 3. Daily cumulative exceeds CTR limit ----------------------------
    if daily_total >= CBN_DAILY_CUMULATIVE_LIMIT:
        red_flags.append(
            f"Daily cumulative N{daily_total:,.2f} meets/exceeds N10M "
            "CTR threshold -- CTR filing mandatory"
        )

    # -- 4. Structuring / smurfing detection ------------------------------
    structuring_limit = CBN_SINGLE_TRANSACTION_LIMIT * STRUCTURING_AMOUNT_PROXIMITY
    if (
        daily_transaction_count >= STRUCTURING_COUNT_THRESHOLD
        and amount >= structuring_limit
    ):
        red_flags.append(
            f"Possible structuring: {daily_transaction_count} transactions "
            f"today, each near N{structuring_limit:,.2f} threshold"
        )

    # -- 5. Unusual hours -------------------------------------------------
    if _ODD_HOUR_START <= hour_of_day < _ODD_HOUR_END:
        red_flags.append(
            f"Transaction at {hour_of_day:02d}:00 WAT -- unusual hour "
            "(midnight-05:00)"
        )

    # -- 6. Geographic mismatch -------------------------------------------
    if (
        customer_state
        and transaction_state
        and customer_state.lower() != transaction_state.lower()
    ):
        red_flags.append(
            f"Geographic mismatch: customer registered in {customer_state}, "
            f"transaction originated in {transaction_state}"
        )

    # -- 7. High-risk channel at odd hours --------------------------------
    channel_lower = channel.lower() if channel else ""
    if channel_lower in _HIGH_RISK_CHANNELS and (
        _ODD_HOUR_START <= hour_of_day < _ODD_HOUR_END
    ):
        red_flags.append(
            f"High-risk channel ({channel}) used during unusual hours"
        )

    # -- Derive risk level using the categorisation function ---------------
    risk_result = categorize_transaction_risk(
        amount=amount,
        channel=channel,
        hour_of_day=hour_of_day,
        daily_total=daily_total,
        fraud_score=fraud_score,
    )
    risk_level: RiskLevel = risk_result["risk_level"]

    # -- Decision logic ---------------------------------------------------
    # File STR if:
    #   (a) 2+ red flags, OR
    #   (b) risk level is CRITICAL, OR
    #   (c) fraud score >= 0.9 (strong model signal), OR
    #   (d) structuring indicators present
    file_str = (
        len(red_flags) >= 2
        or risk_level == RiskLevel.CRITICAL
        or fraud_score >= 0.9
    )

    if file_str:
        recommendation = (
            "IMMEDIATE ACTION REQUIRED: File a Suspicious Transaction Report "
            "with the NFIU within 24 hours per Section 6 of the Money "
            "Laundering (Prohibition) Act 2011.  Preserve all related "
            "records and do not tip off the customer (Section 15)."
        )
    elif red_flags:
        recommendation = (
            "ELEVATED RISK: One or more red-flag indicators detected.  "
            "Escalate to Compliance Officer for manual review.  Consider "
            "enhanced monitoring for this customer."
        )
    else:
        recommendation = "No STR indicators detected. Continue normal monitoring."

    return {
        "file_str": file_str,
        "risk_level": risk_level,
        "red_flags": red_flags,
        "recommendation": recommendation,
    }
