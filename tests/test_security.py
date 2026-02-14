"""
Tests for security middleware and Paystack webhook signature verification:
  - api/middleware/security.py  (_is_exempt, _check_rate_limit)
  - integrations/paystack/webhooks.py  (verify_paystack_signature)
"""

import sys
import os
import hashlib
import hmac
import time

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.middleware.security import (
    _is_exempt,
    _check_rate_limit,
    _request_log,
    RATE_LIMIT_MAX_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
    EXEMPT_PATHS,
    EXEMPT_PREFIXES,
)
from integrations.paystack.webhooks import (
    verify_paystack_signature,
    extract_transaction_data,
)


# ======================================================================
# 1. Path exemption checks
# ======================================================================

class TestIsExempt:
    def test_health_is_exempt(self):
        assert _is_exempt("/health") is True

    def test_docs_is_exempt(self):
        assert _is_exempt("/docs") is True

    def test_openapi_json_is_exempt(self):
        assert _is_exempt("/openapi.json") is True

    def test_redoc_is_exempt(self):
        assert _is_exempt("/redoc") is True

    def test_paystack_webhook_prefix_is_exempt(self):
        assert _is_exempt("/webhooks/paystack") is True

    def test_paystack_webhook_subpath_is_exempt(self):
        assert _is_exempt("/webhooks/paystack/callback") is True

    def test_transactions_not_exempt(self):
        assert _is_exempt("/transactions") is False

    def test_alerts_not_exempt(self):
        assert _is_exempt("/alerts") is False

    def test_dashboard_not_exempt(self):
        assert _is_exempt("/dashboard/stats") is False

    def test_root_not_exempt(self):
        assert _is_exempt("/") is False


# ======================================================================
# 2. Rate limiting
# ======================================================================

class TestCheckRateLimit:
    def setup_method(self):
        """Clear the global request log before each test."""
        _request_log.clear()

    def test_first_request_allowed(self):
        result = _check_rate_limit("test-key-1")
        assert result is None  # None means allowed

    def test_within_limit_allowed(self):
        """Many requests within the limit should all be allowed."""
        for _ in range(RATE_LIMIT_MAX_REQUESTS - 1):
            result = _check_rate_limit("test-key-2")
            assert result is None

    def test_exceeding_limit_returns_429(self):
        """Requests beyond the limit should be rejected with 429."""
        key = "test-key-3"
        # Fill up to the limit
        for _ in range(RATE_LIMIT_MAX_REQUESTS):
            _check_rate_limit(key)

        # Next request should be rejected
        result = _check_rate_limit(key)
        assert result is not None
        assert result.status_code == 429

    def test_different_keys_independent(self):
        """Rate limits should be per-key."""
        for _ in range(RATE_LIMIT_MAX_REQUESTS):
            _check_rate_limit("key-A")

        # key-A should be limited
        assert _check_rate_limit("key-A") is not None

        # key-B should still be allowed
        assert _check_rate_limit("key-B") is None

    def test_old_requests_expire(self):
        """Requests older than the window should be pruned."""
        key = "test-key-4"
        old_time = time.time() - RATE_LIMIT_WINDOW_SECONDS - 10

        # Manually insert old timestamps
        _request_log[key] = [old_time] * RATE_LIMIT_MAX_REQUESTS

        # A new request should succeed because old ones are expired
        result = _check_rate_limit(key)
        assert result is None


# ======================================================================
# 3. Paystack webhook signature verification
# ======================================================================

class TestVerifyPaystackSignature:
    SECRET = "whsec_test_secret_key_12345"

    def _compute_signature(self, payload: bytes) -> str:
        """Helper to compute a valid HMAC-SHA512 signature."""
        return hmac.new(
            key=self.SECRET.encode("utf-8"),
            msg=payload,
            digestmod=hashlib.sha512,
        ).hexdigest()

    def test_valid_signature(self):
        payload = b'{"event": "charge.success", "data": {}}'
        sig = self._compute_signature(payload)
        assert verify_paystack_signature(payload, sig, self.SECRET) is True

    def test_invalid_signature(self):
        payload = b'{"event": "charge.success"}'
        assert verify_paystack_signature(payload, "invalidsig", self.SECRET) is False

    def test_tampered_payload(self):
        """If the payload is modified after signing, verification should fail."""
        original = b'{"event": "charge.success"}'
        sig = self._compute_signature(original)
        tampered = b'{"event": "charge.failed"}'
        assert verify_paystack_signature(tampered, sig, self.SECRET) is False

    def test_empty_payload(self):
        assert verify_paystack_signature(b"", "somesig", self.SECRET) is False

    def test_empty_signature(self):
        assert verify_paystack_signature(b"data", "", self.SECRET) is False

    def test_empty_secret(self):
        assert verify_paystack_signature(b"data", "sig", "") is False

    def test_none_inputs(self):
        assert verify_paystack_signature(None, "sig", self.SECRET) is False
        assert verify_paystack_signature(b"data", None, self.SECRET) is False
        assert verify_paystack_signature(b"data", "sig", None) is False


# ======================================================================
# 4. Paystack transaction data extraction
# ======================================================================

class TestExtractTransactionData:
    def test_basic_extraction(self):
        webhook_data = {
            "event": "charge.success",
            "data": {
                "reference": "PAY_ref_123",
                "amount": 500000,  # 5000 NGN in kobo
                "channel": "card",
                "customer": {
                    "email": "test@example.com",
                    "phone": "+2348012345678",
                },
                "metadata": {},
            },
        }
        result = extract_transaction_data(webhook_data)
        assert result["paystack_ref"] == "PAY_ref_123"
        assert result["amount_ngn"] == 5000.0
        assert result["channel"] == "card"
        assert result["customer_id"] != "unknown"
        assert result["customer_phone_hash"] is not None

    def test_kobo_to_naira_conversion(self):
        webhook_data = {
            "data": {
                "amount": 100,  # 1 NGN
                "reference": "ref",
                "channel": "ussd",
                "customer": {"email": "a@b.com"},
            },
        }
        result = extract_transaction_data(webhook_data)
        assert result["amount_ngn"] == 1.0

    def test_channel_mapping(self):
        """Paystack 'bank' channel should map to 'bank_transfer'."""
        webhook_data = {
            "data": {
                "amount": 10000,
                "reference": "ref",
                "channel": "bank",
                "customer": {"email": "a@b.com"},
            },
        }
        result = extract_transaction_data(webhook_data)
        assert result["channel"] == "bank_transfer"

    def test_missing_customer_email(self):
        """Missing email should produce customer_id='unknown'."""
        webhook_data = {
            "data": {
                "amount": 10000,
                "reference": "ref",
                "channel": "card",
                "customer": {},
            },
        }
        result = extract_transaction_data(webhook_data)
        assert result["customer_id"] == "unknown"

    def test_metadata_location_extraction(self):
        webhook_data = {
            "data": {
                "amount": 10000,
                "reference": "ref",
                "channel": "pos",
                "customer": {"email": "x@y.com"},
                "metadata": {"location_state": "Lagos"},
            },
        }
        result = extract_transaction_data(webhook_data)
        assert result["location_state"] == "Lagos"
