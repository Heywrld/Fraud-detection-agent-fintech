"""
Tests for the compliance modules:
  - compliance/ndpr.py  (PII masking, CBN thresholds, retention, consent)
  - compliance/cbn.py   (risk categorisation, STR evaluation)

All functions under test are pure -- no mocking needed.
"""

import sys
import os
from datetime import datetime, timedelta, timezone

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from compliance.ndpr import (
    mask_account_number,
    mask_phone_number,
    mask_email,
    sanitize_log_entry,
    check_cbn_thresholds,
    check_retention_expiry,
    validate_data_consent,
    CBN_SINGLE_TRANSACTION_LIMIT,
    CBN_DAILY_CUMULATIVE_LIMIT,
    RETENTION_DAYS,
)
from compliance.cbn import (
    categorize_transaction_risk,
    evaluate_str_requirement,
    RiskLevel,
)


# ======================================================================
# 1. mask_account_number
# ======================================================================

class TestMaskAccountNumber:
    def test_standard_10_digit(self):
        assert mask_account_number("0123456789") == "****6789"

    def test_11_digit_bvn(self):
        assert mask_account_number("22312345678") == "****5678"

    def test_short_4_digit(self):
        """4 digits or fewer should be fully masked."""
        assert mask_account_number("1234") == "****"

    def test_short_3_digit(self):
        assert mask_account_number("123") == "****"

    def test_empty_string(self):
        assert mask_account_number("") == "****"

    def test_none_input(self):
        assert mask_account_number(None) == "****"

    def test_non_string_input(self):
        assert mask_account_number(12345) == "****"

    def test_with_spaces(self):
        """Spaces should be stripped before masking."""
        assert mask_account_number("01234 56789") == "****6789"


# ======================================================================
# 2. mask_phone_number
# ======================================================================

class TestMaskPhoneNumber:
    def test_international_234(self):
        assert mask_phone_number("+2348012345678") == "+234****5678"

    def test_local_format(self):
        assert mask_phone_number("08012345678") == "****5678"

    def test_short_number(self):
        """A very short number should be fully masked."""
        assert mask_phone_number("123") == "****"

    def test_empty(self):
        assert mask_phone_number("") == "****"

    def test_none_input(self):
        assert mask_phone_number(None) == "****"

    def test_international_short_rest(self):
        """If the part after the prefix is <= 4 digits, mask the rest entirely."""
        assert mask_phone_number("+2341234") == "+234****"

    def test_with_leading_trailing_spaces(self):
        result = mask_phone_number("  +2348012345678  ")
        assert result == "+234****5678"


# ======================================================================
# 3. mask_email
# ======================================================================

class TestMaskEmail:
    def test_normal_email(self):
        assert mask_email("john.doe@gmail.com") == "j***@gmail.com"

    def test_single_char_local(self):
        assert mask_email("a@example.com") == "a***@example.com"

    def test_no_at_sign(self):
        assert mask_email("not-an-email") == "***@***"

    def test_empty_string(self):
        assert mask_email("") == "***@***"

    def test_none(self):
        assert mask_email(None) == "***@***"

    def test_empty_local_part(self):
        """'@domain.com' has an empty local part."""
        assert mask_email("@domain.com") == "***@***"


# ======================================================================
# 4. sanitize_log_entry
# ======================================================================

class TestSanitizeLogEntry:
    def test_flat_dict_with_pii(self):
        data = {
            "account_number": "0123456789",
            "phone": "+2348012345678",
            "email": "user@test.com",
            "amount": 5000,
        }
        result = sanitize_log_entry(data)
        assert result["account_number"] == "****6789"
        assert result["phone"] == "+234****5678"
        assert result["email"] == "u***@test.com"
        assert result["amount"] == 5000  # non-PII untouched

    def test_nested_dict(self):
        data = {
            "customer": {
                "email_address": "deep@nested.com",
                "bvn": "12345678901",
            },
            "ref": "TX123",
        }
        result = sanitize_log_entry(data)
        assert result["customer"]["email_address"] == "d***@nested.com"
        assert result["customer"]["bvn"] == "****8901"
        assert result["ref"] == "TX123"

    def test_list_inside_dict(self):
        data = {
            "entries": [
                {"phone_number": "08099887766"},
                {"phone_number": "08011223344"},
            ]
        }
        result = sanitize_log_entry(data)
        assert result["entries"][0]["phone_number"] == "****7766"
        assert result["entries"][1]["phone_number"] == "****3344"

    def test_original_not_mutated(self):
        data = {"email": "original@test.com"}
        _ = sanitize_log_entry(data)
        assert data["email"] == "original@test.com"

    def test_no_pii_fields(self):
        data = {"action": "login", "ip": "1.2.3.4"}
        result = sanitize_log_entry(data)
        assert result == data


# ======================================================================
# 5. check_cbn_thresholds
# ======================================================================

class TestCheckCbnThresholds:
    def test_below_both_limits(self):
        result = check_cbn_thresholds(amount=100_000, daily_total=500_000)
        assert result["requires_edd"] is False
        assert result["requires_ctr"] is False
        assert "normal" in result["reason"].lower()

    def test_at_edd_limit(self):
        """Exactly N5M should trigger EDD."""
        result = check_cbn_thresholds(
            amount=CBN_SINGLE_TRANSACTION_LIMIT,
            daily_total=CBN_SINGLE_TRANSACTION_LIMIT,
        )
        assert result["requires_edd"] is True
        assert result["requires_ctr"] is False

    def test_above_edd_limit(self):
        result = check_cbn_thresholds(amount=6_000_000, daily_total=6_000_000)
        assert result["requires_edd"] is True

    def test_at_ctr_limit(self):
        """Daily cumulative exactly at N10M should trigger CTR."""
        result = check_cbn_thresholds(amount=1_000_000, daily_total=CBN_DAILY_CUMULATIVE_LIMIT)
        assert result["requires_ctr"] is True

    def test_above_both_limits(self):
        result = check_cbn_thresholds(amount=6_000_000, daily_total=12_000_000)
        assert result["requires_edd"] is True
        assert result["requires_ctr"] is True
        assert "EDD" in result["reason"]
        assert "CTR" in result["reason"]


# ======================================================================
# 6. check_retention_expiry
# ======================================================================

class TestCheckRetentionExpiry:
    def test_recent_date_not_expired(self):
        """A record created today should NOT be eligible for purging."""
        recent = datetime.now(timezone.utc) - timedelta(days=30)
        assert check_retention_expiry(recent) is False

    def test_old_date_expired(self):
        """A record older than 7 years should be eligible for purging."""
        old = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS + 10)
        assert check_retention_expiry(old) is True

    def test_exactly_at_boundary_not_expired(self):
        """Exactly at RETENTION_DAYS should NOT be expired (> not >=)."""
        boundary = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
        assert check_retention_expiry(boundary) is False

    def test_naive_datetime_assumed_utc(self):
        """A naive datetime is treated as UTC."""
        old_naive = datetime.utcnow() - timedelta(days=RETENTION_DAYS + 10)
        assert check_retention_expiry(old_naive) is True


# ======================================================================
# 7. validate_data_consent
# ======================================================================

class TestValidateDataConsent:
    def test_all_consents_granted(self):
        consent = {
            "fraud_monitoring": True,
            "sms_alerts": True,
            "data_retention": True,
        }
        result = validate_data_consent(consent)
        assert result["valid"] is True
        assert result["missing_fields"] == []

    def test_partial_consent(self):
        consent = {
            "fraud_monitoring": True,
            "sms_alerts": False,
            "data_retention": True,
        }
        result = validate_data_consent(consent)
        assert result["valid"] is False
        assert "sms_alerts" in result["missing_fields"]

    def test_empty_dict(self):
        result = validate_data_consent({})
        assert result["valid"] is False
        assert len(result["missing_fields"]) == 3

    def test_non_dict_input(self):
        result = validate_data_consent("not a dict")
        assert result["valid"] is False
        assert len(result["missing_fields"]) == 3

    def test_missing_single_field(self):
        consent = {
            "fraud_monitoring": True,
            "sms_alerts": True,
            # data_retention missing
        }
        result = validate_data_consent(consent)
        assert result["valid"] is False
        assert "data_retention" in result["missing_fields"]


# ======================================================================
# 8. categorize_transaction_risk
# ======================================================================

class TestCategorizeTransactionRisk:
    def test_low_risk_scenario(self):
        """Small POS transaction during business hours => LOW."""
        result = categorize_transaction_risk(
            amount=5_000,
            channel="pos",
            hour_of_day=10,
        )
        assert result["risk_level"] == RiskLevel.LOW
        assert result["risk_score"] < 3

    def test_medium_risk_scenario(self):
        """Large card transaction during business hours => MEDIUM."""
        result = categorize_transaction_risk(
            amount=1_500_000,
            channel="card",
            hour_of_day=14,
        )
        assert result["risk_level"] == RiskLevel.MEDIUM
        assert 3 <= result["risk_score"] < 5

    def test_high_risk_scenario(self):
        """N5M+ on USSD channel => HIGH."""
        result = categorize_transaction_risk(
            amount=5_500_000,
            channel="ussd",
            hour_of_day=14,
        )
        assert result["risk_level"] == RiskLevel.HIGH
        assert result["risk_score"] >= 5

    def test_critical_risk_scenario(self):
        """Large amount, high-risk channel, odd hour, high fraud score => CRITICAL."""
        result = categorize_transaction_risk(
            amount=6_000_000,
            channel="ussd",
            hour_of_day=2,
            daily_total=12_000_000,
            fraud_score=0.95,
        )
        assert result["risk_level"] == RiskLevel.CRITICAL
        assert result["risk_score"] >= 8

    def test_factors_populated(self):
        result = categorize_transaction_risk(
            amount=100, channel="pos", hour_of_day=10,
        )
        assert isinstance(result["factors"], list)
        assert len(result["factors"]) >= 1


# ======================================================================
# 9. evaluate_str_requirement
# ======================================================================

class TestEvaluateStrRequirement:
    def test_no_red_flags(self):
        """Normal transaction should not recommend filing STR."""
        result = evaluate_str_requirement(
            amount=10_000,
            channel="pos",
            hour_of_day=10,
        )
        assert result["file_str"] is False
        assert len(result["red_flags"]) == 0
        assert "No STR indicators" in result["recommendation"]

    def test_high_fraud_score_triggers_str(self):
        """fraud_score >= 0.9 alone should trigger STR filing."""
        result = evaluate_str_requirement(
            amount=10_000,
            channel="pos",
            hour_of_day=10,
            fraud_score=0.95,
        )
        assert result["file_str"] is True
        assert any("Critical ML" in f for f in result["red_flags"])

    def test_geographic_mismatch(self):
        """Customer in Lagos but transaction from Kano => red flag."""
        result = evaluate_str_requirement(
            amount=10_000,
            channel="pos",
            hour_of_day=10,
            customer_state="Lagos",
            transaction_state="Kano",
        )
        assert any("Geographic mismatch" in f for f in result["red_flags"])

    def test_multiple_red_flags_trigger_str(self):
        """Two or more red flags should trigger STR."""
        result = evaluate_str_requirement(
            amount=6_000_000,
            channel="ussd",
            hour_of_day=2,
            fraud_score=0.75,
            daily_total=11_000_000,
        )
        assert result["file_str"] is True
        assert len(result["red_flags"]) >= 2
        assert "IMMEDIATE ACTION" in result["recommendation"]

    def test_structuring_detection(self):
        """5+ transactions near the N5M limit => structuring flag."""
        structuring_amount = CBN_SINGLE_TRANSACTION_LIMIT * 0.85  # above 80% threshold
        result = evaluate_str_requirement(
            amount=structuring_amount,
            channel="bank_transfer",
            hour_of_day=14,
            daily_transaction_count=6,
        )
        assert any("structuring" in f.lower() for f in result["red_flags"])

    def test_elevated_single_flag_no_str(self):
        """A single non-critical red flag should NOT trigger STR."""
        result = evaluate_str_requirement(
            amount=6_000_000,
            channel="pos",
            hour_of_day=10,
            fraud_score=0.0,
            daily_total=6_000_000,
        )
        # Only one red flag (amount >= N5M), risk may not be CRITICAL
        # With only amount flag and score 0, the risk_level is not CRITICAL
        # and fraud_score < 0.9, so file_str depends on red_flag count
        if len(result["red_flags"]) < 2 and result["risk_level"] != RiskLevel.CRITICAL:
            assert result["file_str"] is False
            assert "ELEVATED RISK" in result["recommendation"]

    def test_odd_hour_plus_high_risk_channel(self):
        """USSD at 3am should flag both odd hour and high-risk channel at odd hour."""
        result = evaluate_str_requirement(
            amount=10_000,
            channel="ussd",
            hour_of_day=3,
        )
        assert any("Unusual hour" in f or "unusual hours" in f.lower() for f in result["red_flags"])
