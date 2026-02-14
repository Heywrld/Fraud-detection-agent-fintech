"""
Webhook endpoints for payment provider integrations.

Handles incoming webhook events from Paystack, verifies their
authenticity via HMAC-SHA512 signature validation, and routes
charge-success events into the fraud detection pipeline.
"""

import logging
from typing import Union

from fastapi import APIRouter, Request, Response, status

from config import get_settings
from integrations.paystack.webhooks import (
    verify_paystack_signature,
    extract_transaction_data,
)
from models.transaction import TransactionCreate
from services.fraud_engine import process_transaction

logger = logging.getLogger("fraud_guardian.webhooks")

router = APIRouter()


@router.post("/paystack", status_code=status.HTTP_200_OK)
async def paystack_webhook(request: Request) -> Union[dict, Response]:
    """
    Receive and process Paystack webhook events.

    Paystack signs every webhook payload with HMAC-SHA512 using the
    webhook secret key. This endpoint:

    1. Reads the raw request body and the ``X-Paystack-Signature`` header.
    2. Validates the signature -- rejects the request with 401 if invalid.
    3. Parses the JSON payload and dispatches based on the event type.
    4. For ``charge.success`` events, extracts transaction data, validates
       it against our Pydantic schema, and feeds it into the fraud engine.
    5. Returns 200 OK as fast as possible (Paystack expects quick replies).

    All errors are caught so we never return a 5xx to Paystack, which
    would trigger unnecessary retries.
    """
    settings = get_settings()

    # ------------------------------------------------------------------
    # 1. Read raw body and signature header
    # ------------------------------------------------------------------
    try:
        body: bytes = await request.body()
    except Exception as exc:
        logger.error("Failed to read request body: %s", exc)
        return {"status": "error", "message": "Failed to read request body"}

    signature = request.headers.get("x-paystack-signature", "")

    # ------------------------------------------------------------------
    # 2. Verify webhook signature
    # ------------------------------------------------------------------
    if not verify_paystack_signature(
        payload=body,
        signature=signature,
        secret=settings.paystack_webhook_secret,
    ):
        logger.warning(
            "Paystack webhook signature verification failed (IP: %s)",
            request.client.host if request.client else "unknown",
        )
        return Response(
            content='{"status": "unauthorized", "message": "Invalid signature"}',
            status_code=status.HTTP_401_UNAUTHORIZED,
            media_type="application/json",
        )

    # ------------------------------------------------------------------
    # 3. Parse JSON payload
    # ------------------------------------------------------------------
    try:
        payload: dict = await request.json()
    except Exception as exc:
        logger.error("Malformed JSON in Paystack webhook body: %s", exc)
        return {"status": "error", "message": "Invalid JSON payload"}

    event_type: str = payload.get("event", "")
    logger.info("Paystack webhook received: event=%s", event_type)

    # ------------------------------------------------------------------
    # 4. Route by event type
    # ------------------------------------------------------------------
    if event_type == "charge.success":
        await _handle_charge_success(payload)
    else:
        logger.info(
            "Ignoring Paystack event type '%s' -- no handler registered",
            event_type,
        )

    # ------------------------------------------------------------------
    # 5. Always acknowledge quickly
    # ------------------------------------------------------------------
    return {"status": "received"}


async def _handle_charge_success(payload: dict) -> None:
    """
    Process a ``charge.success`` webhook event through the fraud pipeline.

    Extracts transaction fields from the Paystack payload, validates them
    with the ``TransactionCreate`` Pydantic model, and hands the data off
    to ``fraud_engine.process_transaction`` for scoring, alerting, and
    persistence.

    The fraud engine call is launched as a fire-and-forget background task
    wrapped in its own try/except so that a downstream failure never
    prevents the webhook from returning 200 OK to Paystack.
    """
    try:
        # Use the helper from integrations/paystack/webhooks.py to map
        # Paystack fields to our internal schema dict.
        transaction_data = extract_transaction_data(payload)

        # Validate through the Pydantic model to enforce types, required
        # fields, and value constraints (e.g. amount_ngn > 0).
        tx_model = TransactionCreate(**transaction_data)

        logger.info(
            "Processing charge.success: ref=%s, amount=N%.2f, channel=%s, customer=%s",
            tx_model.paystack_ref,
            tx_model.amount_ngn,
            tx_model.channel.value,
            tx_model.customer_id,
        )

        # Hand off to the fraud engine.  We await it here so errors are
        # caught by the surrounding try/except, but the webhook still
        # returns 200 because the caller catches exceptions too.
        result = await process_transaction(tx_model.model_dump())

        logger.info(
            "Fraud engine result for ref=%s: score=%.4f, status=%s",
            tx_model.paystack_ref,
            result.get("fraud_score", -1),
            result.get("status", "unknown"),
        )

    except Exception as exc:
        # Log the full error but never let it propagate -- Paystack must
        # always receive 200 OK.
        logger.exception(
            "Error processing charge.success event: %s", exc,
        )
