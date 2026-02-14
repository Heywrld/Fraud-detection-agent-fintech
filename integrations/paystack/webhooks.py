"""
Paystack webhook signature verification and data extraction.

Paystack signs webhook payloads with HMAC-SHA512 using the secret key.
The signature is sent in the 'x-paystack-signature' header.
"""

import hashlib
import hmac
import logging
from typing import Optional

logger = logging.getLogger("fraud_guardian.paystack.webhooks")

# Map Paystack channel names to our TransactionChannel enum values
CHANNEL_MAP = {
    "card": "card",
    "bank": "bank_transfer",
    "ussd": "ussd",
    "qr": "card",
    "mobile_money": "mobile_money",
    "bank_transfer": "bank_transfer",
    "dedicated_nuban": "bank_transfer",
}


def verify_paystack_signature(payload: bytes, signature: str, secret: str) -> bool:
    """
    Verify the HMAC-SHA512 signature of a Paystack webhook payload.

    Paystack computes HMAC-SHA512 of the raw request body using the
    webhook secret key, and sends it as the x-paystack-signature header.

    Args:
        payload: Raw request body bytes exactly as received.
        signature: Value of the x-paystack-signature header.
        secret: Paystack webhook secret key.

    Returns:
        True if the signature is valid, False otherwise.
    """
    if not payload or not signature or not secret:
        logger.warning("Missing payload, signature, or secret for verification")
        return False

    try:
        expected = hmac.new(
            key=secret.encode("utf-8"),
            msg=payload,
            digestmod=hashlib.sha512,
        ).hexdigest()

        is_valid = hmac.compare_digest(expected, signature)

        if not is_valid:
            logger.warning("Paystack webhook signature mismatch")
        else:
            logger.debug("Paystack webhook signature verified successfully")

        return is_valid

    except Exception as exc:
        logger.error("Error verifying Paystack signature: %s", exc)
        return False


def extract_transaction_data(webhook_data: dict) -> dict:
    """
    Map Paystack webhook fields to our TransactionCreate schema fields.

    This function takes the 'data' portion of a Paystack webhook event
    and produces a dict matching the TransactionCreate model fields:
        - paystack_ref: Paystack transaction reference
        - amount_ngn: Amount converted from kobo to Naira
        - channel: Mapped to our TransactionChannel enum values
        - customer_id: SHA-256 hash of customer email (pseudonymized)
        - customer_phone_hash: SHA-256 hash of phone if present
        - location_state: Extracted from metadata if available
        - device_fingerprint: Extracted from metadata if available
        - metadata: Original metadata dict

    Args:
        webhook_data: The 'data' dict from a Paystack webhook event.

    Returns:
        Dict ready to be passed to TransactionCreate(**result).
    """
    data = webhook_data.get("data", webhook_data)

    # Extract customer info
    customer = data.get("customer", {})
    customer_email = customer.get("email", "")
    customer_phone = customer.get("phone")

    # Hash customer email for pseudonymized customer ID
    customer_id = hashlib.sha256(
        customer_email.lower().strip().encode("utf-8")
    ).hexdigest() if customer_email else "unknown"

    # Hash phone number if present
    customer_phone_hash = None
    if customer_phone:
        customer_phone_hash = hashlib.sha256(
            customer_phone.strip().encode("utf-8")
        ).hexdigest()

    # Map Paystack channel to our enum
    raw_channel = data.get("channel", "card")
    channel = CHANNEL_MAP.get(raw_channel, "card")

    # Extract location and device info from metadata
    metadata = data.get("metadata", {}) or {}

    # Paystack metadata can contain custom_fields as a list of dicts
    # or direct key-value pairs depending on the merchant's integration
    custom_fields = {}
    if isinstance(metadata.get("custom_fields"), list):
        for field in metadata["custom_fields"]:
            if isinstance(field, dict) and "variable_name" in field:
                custom_fields[field["variable_name"]] = field.get("value")

    location_state = (
        metadata.get("location_state")
        or metadata.get("state")
        or custom_fields.get("location_state")
        or custom_fields.get("state")
    )

    device_fingerprint = (
        metadata.get("device_fingerprint")
        or metadata.get("device_id")
        or custom_fields.get("device_fingerprint")
    )

    # Convert kobo to naira
    amount_kobo = data.get("amount", 0)
    amount_ngn = amount_kobo / 100

    transaction_data = {
        "paystack_ref": data.get("reference", ""),
        "amount_ngn": amount_ngn,
        "channel": channel,
        "customer_id": customer_id,
        "customer_phone_hash": customer_phone_hash,
        "location_state": location_state,
        "device_fingerprint": device_fingerprint,
        "metadata": metadata,
    }

    logger.info(
        "Extracted transaction data: ref=%s, amount=N%.2f, channel=%s, state=%s",
        transaction_data["paystack_ref"],
        transaction_data["amount_ngn"],
        transaction_data["channel"],
        transaction_data["location_state"],
    )

    return transaction_data
