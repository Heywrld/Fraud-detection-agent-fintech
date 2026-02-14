"""
Tests for Pydantic models:
  - models/transaction.py
  - models/alert.py

Validates that schema validation, serialization, enum handling,
and field constraints all behave correctly.
"""

import sys
import os
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.transaction import (
    TransactionCreate,
    TransactionChannel,
    TransactionStatus,
    TransactionResponse,
    TransactionListResponse,
)
from models.alert import (
    AlertCreate,
    AlertType,
    AlertLanguage,
    AlertStatus,
    AlertResponse,
    AlertResolveRequest,
)


# ======================================================================
# 1. TransactionCreate
# ======================================================================

class TestTransactionCreate:
    def test_valid_creation(self):
        tx = TransactionCreate(
            paystack_ref="ref_001",
            amount_ngn=1500.50,
            channel="pos",
            customer_id="cust_abc123",
        )
        assert tx.paystack_ref == "ref_001"
        assert tx.amount_ngn == 1500.50
        assert tx.channel == TransactionChannel.POS
        assert tx.customer_id == "cust_abc123"
        assert tx.metadata == {}

    def test_invalid_amount_zero(self):
        """amount_ngn must be > 0 (gt=0)."""
        with pytest.raises(ValidationError) as exc_info:
            TransactionCreate(
                paystack_ref="ref_002",
                amount_ngn=0,
                channel="pos",
                customer_id="cust_abc",
            )
        assert "amount_ngn" in str(exc_info.value)

    def test_invalid_amount_negative(self):
        with pytest.raises(ValidationError):
            TransactionCreate(
                paystack_ref="ref_003",
                amount_ngn=-100,
                channel="pos",
                customer_id="cust_abc",
            )

    def test_optional_fields_default_none(self):
        tx = TransactionCreate(
            paystack_ref="ref_004",
            amount_ngn=100,
            channel="card",
            customer_id="cust_xyz",
        )
        assert tx.customer_phone_hash is None
        assert tx.location_state is None
        assert tx.device_fingerprint is None

    def test_metadata_default_empty_dict(self):
        tx = TransactionCreate(
            paystack_ref="ref_005",
            amount_ngn=100,
            channel="ussd",
            customer_id="cust_xyz",
        )
        assert tx.metadata == {}

    def test_metadata_with_data(self):
        tx = TransactionCreate(
            paystack_ref="ref_006",
            amount_ngn=100,
            channel="bank_transfer",
            customer_id="cust_xyz",
            metadata={"source": "web", "ip": "1.2.3.4"},
        )
        assert tx.metadata["source"] == "web"


# ======================================================================
# 2. TransactionChannel enum
# ======================================================================

class TestTransactionChannel:
    def test_valid_channels(self):
        for ch in ["pos", "mobile_money", "ussd", "bank_transfer", "card"]:
            assert TransactionChannel(ch).value == ch

    def test_invalid_channel(self):
        with pytest.raises(ValueError):
            TransactionChannel("crypto")

    def test_channel_from_model(self):
        tx = TransactionCreate(
            paystack_ref="ref_ch",
            amount_ngn=100,
            channel="mobile_money",
            customer_id="cust",
        )
        assert tx.channel == TransactionChannel.MOBILE_MONEY


# ======================================================================
# 3. TransactionResponse serialization
# ======================================================================

class TestTransactionResponse:
    def test_full_serialization(self):
        now = datetime.now(timezone.utc)
        resp = TransactionResponse(
            id="tx-uuid-1",
            paystack_ref="ref_resp_1",
            amount_ngn=25000.0,
            channel="card",
            customer_id="cust_123",
            location_state="Lagos",
            fraud_score=0.42,
            is_flagged=False,
            status="pending",
            created_at=now,
        )
        data = resp.model_dump()
        assert data["id"] == "tx-uuid-1"
        assert data["fraud_score"] == 0.42
        assert data["status"] == TransactionStatus.PENDING

    def test_list_response(self):
        now = datetime.now(timezone.utc)
        tx_resp = TransactionResponse(
            id="tx-1", paystack_ref="ref_1", amount_ngn=1000,
            channel="pos", customer_id="c1", fraud_score=0.1,
            is_flagged=False, status="pending", created_at=now,
        )
        list_resp = TransactionListResponse(
            transactions=[tx_resp],
            total=1,
            page=1,
            page_size=20,
        )
        assert len(list_resp.transactions) == 1
        assert list_resp.total == 1


# ======================================================================
# 4. AlertResponse model
# ======================================================================

class TestAlertResponse:
    def test_full_serialization(self):
        now = datetime.now(timezone.utc)
        alert = AlertResponse(
            id="alert-uuid-1",
            transaction_id="tx-uuid-1",
            alert_type="sms",
            language="en",
            message="Suspicious activity detected",
            status="pending",
            sent_at=None,
            resolved_at=None,
            resolved_by=None,
            created_at=now,
        )
        data = alert.model_dump()
        assert data["id"] == "alert-uuid-1"
        assert data["alert_type"] == AlertType.SMS
        assert data["language"] == AlertLanguage.ENGLISH
        assert data["status"] == AlertStatus.PENDING
        assert data["sent_at"] is None

    def test_resolved_alert(self):
        now = datetime.now(timezone.utc)
        alert = AlertResponse(
            id="alert-uuid-2",
            transaction_id="tx-uuid-2",
            alert_type="freeze",
            language="ha",
            message="Account frozen",
            status="resolved",
            resolved_at=now,
            resolved_by="admin@example.com",
            created_at=now,
        )
        assert alert.status == AlertStatus.RESOLVED
        assert alert.resolved_by == "admin@example.com"

    def test_all_alert_types(self):
        for at in ["sms", "freeze", "manual_review"]:
            assert AlertType(at).value == at

    def test_all_alert_languages(self):
        for lang in ["en", "ha", "yo", "pcm"]:
            assert AlertLanguage(lang).value == lang

    def test_all_alert_statuses(self):
        for st in ["pending", "sent", "failed", "resolved"]:
            assert AlertStatus(st).value == st


# ======================================================================
# 5. AlertResolveRequest
# ======================================================================

class TestAlertResolveRequest:
    def test_valid_request(self):
        req = AlertResolveRequest(resolved_by="officer_jane")
        assert req.resolved_by == "officer_jane"

    def test_missing_resolved_by(self):
        """resolved_by is required -- omitting it should raise."""
        with pytest.raises(ValidationError):
            AlertResolveRequest()

    def test_empty_string_allowed(self):
        """Pydantic does not forbid empty strings by default."""
        req = AlertResolveRequest(resolved_by="")
        assert req.resolved_by == ""


# ======================================================================
# 6. AlertCreate
# ======================================================================

class TestAlertCreate:
    def test_valid_creation(self):
        alert = AlertCreate(
            transaction_id="tx-1",
            alert_type="sms",
            message="Test alert message",
        )
        assert alert.language == AlertLanguage.ENGLISH  # default
        assert alert.alert_type == AlertType.SMS

    def test_invalid_alert_type(self):
        with pytest.raises(ValidationError):
            AlertCreate(
                transaction_id="tx-1",
                alert_type="invalid_type",
                message="Test",
            )
