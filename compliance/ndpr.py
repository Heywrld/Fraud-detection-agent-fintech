"""
NDPR (Nigeria Data Protection Regulation) and CBN AML/CFT compliance utilities.

This module implements PII masking, CBN AML threshold checks, data-retention
policy enforcement, and NDPR consent validation.  All functions are *pure*
-- no database or network calls -- so they can be unit-tested in isolation
and composed freely by the services layer.

References
----------
- NDPR 2019 (Nigeria Data Protection Regulation)
  https://nitda.gov.ng/wp-content/uploads/2019/01/Nigeria%20Data%20Protection%20Regulation.pdf
- CBN AML/CFT Regulations 2013 (amended 2022)
- CBN Circular: Threshold for Reporting Cash-Based Transactions (CTR) -- N5,000,000 single / N10,000,000 cumulative daily
- Money Laundering (Prohibition) Act 2011 (as amended 2022)
"""

from __future__ import annotations

import copy
import re
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# PII field names that must be masked when they appear in log entries,
# API responses, or audit records.
# ---------------------------------------------------------------------------

_PII_FIELD_PATTERNS: dict[str, str] = {
    # field name (lowercased)  ->  masking strategy
    "account_number": "account",
    "account_no": "account",
    "phone": "phone",
    "phone_number": "phone",
    "customer_phone": "phone",
    "mobile": "phone",
    "msisdn": "phone",
    "email": "email",
    "email_address": "email",
    "bvn": "account",           # Bank Verification Number -- treat like account
    "nin": "account",           # National Identification Number
    "customer_phone_hash": "phone",
}


# ============================================================================
# 1. PII Masking
# ============================================================================

def mask_account_number(account: str) -> str:
    """Mask an account/BVN/NIN showing only the last 4 digits.

    Examples
    --------
    >>> mask_account_number("0123456789")
    '****6789'
    >>> mask_account_number("22312345678")  # 11-digit BVN
    '****5678'
    """
    if not account or not isinstance(account, str):
        return "****"
    clean = re.sub(r"\s+", "", account)
    if len(clean) <= 4:
        return "****"
    return "****" + clean[-4:]


def mask_phone_number(phone: str) -> str:
    """Mask a phone number showing only the last 4 digits.

    Preserves the international prefix if present.

    Examples
    --------
    >>> mask_phone_number("+2348012345678")
    '+234****5678'
    >>> mask_phone_number("08012345678")
    '****5678'
    """
    if not phone or not isinstance(phone, str):
        return "****"
    clean = phone.strip()

    # Detect and preserve international prefix (e.g., +234, +1)
    prefix_match = re.match(r"^(\+\d{1,3})", clean)
    if prefix_match:
        prefix = prefix_match.group(1)
        rest = clean[len(prefix):]
        if len(rest) <= 4:
            return prefix + "****"
        return prefix + "****" + rest[-4:]

    # No prefix -- just mask everything except last 4
    digits = re.sub(r"\D", "", clean)
    if len(digits) <= 4:
        return "****"
    return "****" + digits[-4:]


def mask_email(email: str) -> str:
    """Mask an email address: first character + '***' + domain.

    Examples
    --------
    >>> mask_email("john.doe@gmail.com")
    'j***@gmail.com'
    >>> mask_email("a@example.com")
    'a***@example.com'
    """
    if not email or not isinstance(email, str):
        return "***@***"
    parts = email.split("@", 1)
    if len(parts) != 2 or not parts[0]:
        return "***@***"
    local, domain = parts
    return local[0] + "***@" + domain


def sanitize_log_entry(data: dict) -> dict:
    """Deep-copy *data* and mask every PII field found at any nesting level.

    The function walks dicts and lists recursively.  Field names are matched
    case-insensitively against ``_PII_FIELD_PATTERNS``.

    Parameters
    ----------
    data : dict
        Arbitrary log / audit payload that may contain PII.

    Returns
    -------
    dict
        A new dict with all recognised PII fields replaced by masked values.
    """
    return _sanitize_recursive(copy.deepcopy(data))


def _sanitize_recursive(obj: Any) -> Any:
    """Recursively walk a structure and mask PII fields."""
    if isinstance(obj, dict):
        sanitized: dict[str, Any] = {}
        for key, value in obj.items():
            strategy = _PII_FIELD_PATTERNS.get(key.lower())
            if strategy and isinstance(value, str):
                if strategy == "account":
                    sanitized[key] = mask_account_number(value)
                elif strategy == "phone":
                    sanitized[key] = mask_phone_number(value)
                elif strategy == "email":
                    sanitized[key] = mask_email(value)
                else:
                    sanitized[key] = "****"
            else:
                sanitized[key] = _sanitize_recursive(value)
        return sanitized
    elif isinstance(obj, list):
        return [_sanitize_recursive(item) for item in obj]
    return obj


# ============================================================================
# 2. CBN AML Thresholds
# ============================================================================

# CBN Circular on Cash-Based Transaction Reporting, 2022 revision.
# Single transaction >= N5,000,000 requires Enhanced Due Diligence (EDD).
CBN_SINGLE_TRANSACTION_LIMIT: float = 5_000_000  # Naira

# Daily cumulative >= N10,000,000 triggers a Currency Transaction Report (CTR)
# filing with the NFIU.
CBN_DAILY_CUMULATIVE_LIMIT: float = 10_000_000  # Naira


def check_cbn_thresholds(amount: float, daily_total: float) -> dict:
    """Evaluate a transaction against CBN AML cash-reporting thresholds.

    Parameters
    ----------
    amount : float
        The value of the *current* transaction in Naira.
    daily_total : float
        The sum of all transactions for this customer on the current
        calendar day, **including** the current transaction.

    Returns
    -------
    dict
        ``requires_edd``  -- True if Enhanced Due Diligence is needed
        (single txn >= N5M).
        ``requires_ctr``  -- True if a Currency Transaction Report must
        be filed with NFIU (daily cumulative >= N10M).
        ``reason``        -- Human-readable explanation.
    """
    requires_edd = amount >= CBN_SINGLE_TRANSACTION_LIMIT
    requires_ctr = daily_total >= CBN_DAILY_CUMULATIVE_LIMIT

    reasons: list[str] = []
    if requires_edd:
        reasons.append(
            f"Single transaction N{amount:,.2f} meets/exceeds CBN EDD "
            f"threshold of N{CBN_SINGLE_TRANSACTION_LIMIT:,.2f}"
        )
    if requires_ctr:
        reasons.append(
            f"Daily cumulative N{daily_total:,.2f} meets/exceeds CBN CTR "
            f"threshold of N{CBN_DAILY_CUMULATIVE_LIMIT:,.2f}"
        )
    if not reasons:
        reasons.append("Transaction within normal CBN limits")

    return {
        "requires_edd": requires_edd,
        "requires_ctr": requires_ctr,
        "reason": "; ".join(reasons),
    }


# ============================================================================
# 3. Data Retention
# ============================================================================

# CBN AML/CFT Regulations, Section 9 -- financial institutions must retain
# records for a minimum of 7 years from the date of the transaction.
RETENTION_DAYS: int = 365 * 7  # 2 555 days


def check_retention_expiry(created_at: datetime) -> bool:
    """Return ``True`` if the record has exceeded the 7-year retention window
    and is eligible for purging.

    Parameters
    ----------
    created_at : datetime
        The creation timestamp of the record.  If naive (no tzinfo), it is
        assumed to be UTC.

    Returns
    -------
    bool
        ``True`` when the record **can** be purged; ``False`` when it must
        be retained.
    """
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    age_days = (datetime.now(timezone.utc) - created_at).days
    return age_days > RETENTION_DAYS


# ============================================================================
# 4. Consent Validation (NDPR)
# ============================================================================

# NDPR requires explicit consent for each specific purpose of data processing.
# For Fraud Guardian the three relevant consent fields are:
#   - fraud_monitoring : permission to score & flag transactions
#   - sms_alerts       : permission to send SMS notifications
#   - data_retention   : permission to store transaction data for 7 years
_REQUIRED_CONSENT_FIELDS: list[str] = [
    "fraud_monitoring",
    "sms_alerts",
    "data_retention",
]


def validate_data_consent(user_consent: dict) -> dict:
    """Check that a user's consent dict contains all required NDPR fields.

    Each required field must be present **and** have a truthy boolean value.

    Parameters
    ----------
    user_consent : dict
        Consent record, e.g.
        ``{"fraud_monitoring": True, "sms_alerts": True, "data_retention": True}``

    Returns
    -------
    dict
        ``valid``          -- True if all fields are present and True.
        ``missing_fields`` -- List of field names that are missing or False.
    """
    if not isinstance(user_consent, dict):
        return {
            "valid": False,
            "missing_fields": list(_REQUIRED_CONSENT_FIELDS),
        }

    missing: list[str] = []
    for field in _REQUIRED_CONSENT_FIELDS:
        value = user_consent.get(field)
        if value is not True:
            missing.append(field)

    return {
        "valid": len(missing) == 0,
        "missing_fields": missing,
    }
