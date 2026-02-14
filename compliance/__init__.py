"""
Fraud Guardian -- NDPR & CBN Compliance Module.

Exports PII masking, CBN AML threshold checks, data-retention helpers,
NDPR consent validation, risk categorisation, and STR flag logic.
"""

# -- NDPR: PII Masking --
from compliance.ndpr import (
    mask_account_number,
    mask_phone_number,
    mask_email,
    sanitize_log_entry,
)

# -- CBN AML Thresholds --
from compliance.ndpr import (
    CBN_SINGLE_TRANSACTION_LIMIT,
    CBN_DAILY_CUMULATIVE_LIMIT,
    check_cbn_thresholds,
)

# -- Data Retention --
from compliance.ndpr import (
    RETENTION_DAYS,
    check_retention_expiry,
)

# -- NDPR Consent --
from compliance.ndpr import validate_data_consent

# -- CBN Risk Categorisation --
from compliance.cbn import (
    RiskLevel,
    categorize_transaction_risk,
)

# -- Suspicious Transaction Report (STR) --
from compliance.cbn import evaluate_str_requirement

__all__ = [
    # PII masking
    "mask_account_number",
    "mask_phone_number",
    "mask_email",
    "sanitize_log_entry",
    # CBN AML thresholds
    "CBN_SINGLE_TRANSACTION_LIMIT",
    "CBN_DAILY_CUMULATIVE_LIMIT",
    "check_cbn_thresholds",
    # Data retention
    "RETENTION_DAYS",
    "check_retention_expiry",
    # NDPR consent
    "validate_data_consent",
    # CBN risk categorisation
    "RiskLevel",
    "categorize_transaction_risk",
    # STR
    "evaluate_str_requirement",
]
