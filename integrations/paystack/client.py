"""
Paystack API client for transaction verification and retrieval.

Uses httpx.AsyncClient to communicate with the Paystack REST API.
All monetary amounts from Paystack are in kobo (1 NGN = 100 kobo).
"""

import httpx
import logging
from typing import Optional

from config import get_settings

logger = logging.getLogger("fraud_guardian.paystack")

PAYSTACK_BASE_URL = "https://api.paystack.co"


class PaystackClient:
    """Async client for the Paystack API."""

    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.base_url = PAYSTACK_BASE_URL
        self._headers = {
            "Authorization": f"Bearer {secret_key}",
            "Content-Type": "application/json",
        }

    async def _request(self, method: str, path: str, params: Optional[dict] = None) -> dict:
        """
        Make an authenticated request to the Paystack API.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path relative to base URL (e.g. /transaction/verify/ref123)
            params: Optional query parameters

        Returns:
            Parsed JSON response as a dict.

        Raises:
            httpx.HTTPStatusError: On 4xx/5xx responses after logging the error.
        """
        url = f"{self.base_url}{path}"
        try:
            async with httpx.AsyncClient(headers=self._headers, timeout=30.0) as client:
                response = await client.request(method, url, params=params)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Paystack API error: %s %s -> %d: %s",
                method,
                url,
                exc.response.status_code,
                exc.response.text,
            )
            raise
        except httpx.RequestError as exc:
            logger.error("Paystack request failed for %s %s: %s", method, url, exc)
            raise

    async def verify_transaction(self, reference: str) -> dict:
        """
        Verify a transaction by its reference.

        GET /transaction/verify/{reference}

        Args:
            reference: The Paystack transaction reference string.

        Returns:
            Full Paystack verification response including status and data.
        """
        logger.info("Verifying transaction reference: %s", reference)
        return await self._request("GET", f"/transaction/verify/{reference}")

    async def list_transactions(self, page: int = 1, per_page: int = 50) -> dict:
        """
        List transactions on the Paystack account.

        GET /transaction

        Args:
            page: Page number (1-indexed).
            per_page: Number of records per page (max 200, default 50).

        Returns:
            Paginated list of transactions with metadata.
        """
        params = {"page": page, "perPage": per_page}
        logger.info("Listing transactions: page=%d, per_page=%d", page, per_page)
        return await self._request("GET", "/transaction", params=params)

    async def get_transaction(self, id: int) -> dict:
        """
        Fetch a single transaction by its Paystack ID.

        GET /transaction/{id}

        Args:
            id: Paystack numeric transaction ID.

        Returns:
            Full transaction details from Paystack.
        """
        logger.info("Fetching transaction ID: %d", id)
        return await self._request("GET", f"/transaction/{id}")


def get_paystack_client() -> PaystackClient:
    """
    Factory function to create a PaystackClient from application settings.

    Reads the Paystack secret key from the config and returns
    a configured client instance.
    """
    settings = get_settings()
    return PaystackClient(secret_key=settings.paystack_secret_key)
