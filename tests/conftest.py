"""
Shared fixtures for the Fraud Guardian AI test suite.

Provides:
- test_settings: a Settings instance with deterministic defaults
- mock_supabase: patches the Supabase client for tests that need DB isolation
"""

import sys
import os
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on sys.path so imports like `from config import ...` work.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Settings


# ---------------------------------------------------------------------------
# 1. Test Settings fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def test_settings() -> Settings:
    """Return a Settings instance with known, deterministic values.

    These values are safe for testing -- no real secrets or production URLs.
    """
    return Settings(
        app_env="testing",
        api_key="test-api-key-12345",
        supabase_url="https://test.supabase.co",
        supabase_key="test-anon-key",
        supabase_service_key="test-service-key",
        paystack_secret_key="sk_test_xxx",
        paystack_public_key="pk_test_xxx",
        paystack_webhook_secret="whsec_test_secret",
        twilio_account_sid="AC_test",
        twilio_auth_token="test-auth-token",
        twilio_phone_number="+10000000000",
        fraud_score_flag_threshold=0.7,
        fraud_score_freeze_threshold=0.9,
        model_path="ml/artifacts/fraud_model.joblib",
    )


# ---------------------------------------------------------------------------
# 2. Mock Supabase client fixture
# ---------------------------------------------------------------------------

class _FakeQueryBuilder:
    """A chainable mock that mimics the Supabase query-builder pattern.

    Every method call returns ``self`` so that chained calls like
    ``client.table("x").select("*").eq("id", 1).execute()`` resolve
    without errors.  ``execute()`` returns a simple result object.
    """

    def __init__(self, data=None, count=0):
        self._data = data if data is not None else []
        self._count = count

    # Allow arbitrary chaining
    def __getattr__(self, name):
        def _chain(*args, **kwargs):
            return self
        return _chain

    def execute(self):
        result = MagicMock()
        result.data = self._data
        result.count = self._count
        return result


class _FakeSupabaseClient:
    """Minimal stand-in for the Supabase Python client."""

    def __init__(self, data=None, count=0):
        self._data = data if data is not None else []
        self._count = count

    def table(self, table_name: str):
        return _FakeQueryBuilder(data=self._data, count=self._count)


@pytest.fixture()
def mock_supabase():
    """Patch ``db.supabase_client.get_supabase_client`` to return a fake client.

    Yields the fake client instance so tests can inspect or customise it.
    """
    fake_client = _FakeSupabaseClient()

    with patch("db.supabase_client.get_supabase_client", return_value=fake_client):
        yield fake_client
