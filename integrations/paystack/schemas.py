"""
Pydantic schemas for Paystack webhook payloads.

Paystack sends amounts in kobo (100 kobo = 1 NGN). The schemas here
normalize the amount to Naira for downstream consumption.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
import logging

logger = logging.getLogger("fraud_guardian.paystack.schemas")


class PaystackCustomer(BaseModel):
    """Customer data embedded in Paystack charge events."""
    email: Optional[str] = None
    phone: Optional[str] = None
    customer_code: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None


class PaystackAuthorization(BaseModel):
    """Authorization/card data from Paystack charge events."""
    channel: Optional[str] = None
    card_type: Optional[str] = None
    bank: Optional[str] = None
    country_code: Optional[str] = None
    brand: Optional[str] = None
    reusable: Optional[bool] = None
    signature: Optional[str] = None
    bin: Optional[str] = None
    last4: Optional[str] = None
    exp_month: Optional[str] = None
    exp_year: Optional[str] = None


class PaystackChargeData(BaseModel):
    """
    Normalized charge data extracted from a Paystack webhook event.

    The amount field is automatically converted from kobo to Naira.
    """
    id: int
    reference: str
    amount: float = Field(description="Transaction amount in Naira (converted from kobo)")
    amount_kobo: int = Field(description="Original amount in kobo as received from Paystack")
    channel: str
    currency: str = "NGN"
    customer: PaystackCustomer = Field(default_factory=PaystackCustomer)
    metadata: dict = Field(default_factory=dict)
    paid_at: Optional[str] = None
    created_at: Optional[str] = None
    authorization: PaystackAuthorization = Field(default_factory=PaystackAuthorization)
    status: Optional[str] = None
    gateway_response: Optional[str] = None
    ip_address: Optional[str] = None


class PaystackWebhookEvent(BaseModel):
    """
    Top-level Paystack webhook event envelope.

    Paystack sends events with an 'event' field (e.g. 'charge.success')
    and a 'data' dict containing the transaction details.
    """
    event: str
    data: dict


def parse_webhook_event(payload: dict) -> PaystackChargeData:
    """
    Parse a raw Paystack webhook payload into a normalized PaystackChargeData.

    This function:
    1. Validates the top-level webhook structure.
    2. Extracts the 'data' dict.
    3. Converts the amount from kobo to Naira (amount / 100).
    4. Parses customer and authorization sub-objects.

    Args:
        payload: Raw webhook JSON body as a dict.

    Returns:
        PaystackChargeData with the amount in Naira.

    Raises:
        ValueError: If the payload cannot be parsed.
    """
    try:
        # Validate envelope
        event = PaystackWebhookEvent(**payload)
        data = event.data

        # Extract and build customer
        raw_customer = data.get("customer", {})
        customer = PaystackCustomer(
            email=raw_customer.get("email"),
            phone=raw_customer.get("phone"),
            customer_code=raw_customer.get("customer_code"),
            first_name=raw_customer.get("first_name"),
            last_name=raw_customer.get("last_name"),
        )

        # Extract and build authorization
        raw_auth = data.get("authorization", {})
        authorization = PaystackAuthorization(
            channel=raw_auth.get("channel"),
            card_type=raw_auth.get("card_type"),
            bank=raw_auth.get("bank"),
            country_code=raw_auth.get("country_code"),
            brand=raw_auth.get("brand"),
            reusable=raw_auth.get("reusable"),
            signature=raw_auth.get("signature"),
            bin=raw_auth.get("bin"),
            last4=raw_auth.get("last4"),
            exp_month=raw_auth.get("exp_month"),
            exp_year=raw_auth.get("exp_year"),
        )

        # Convert kobo to naira
        amount_kobo = data.get("amount", 0)
        amount_naira = amount_kobo / 100

        charge_data = PaystackChargeData(
            id=data["id"],
            reference=data["reference"],
            amount=amount_naira,
            amount_kobo=amount_kobo,
            channel=data.get("channel", "unknown"),
            currency=data.get("currency", "NGN"),
            customer=customer,
            metadata=data.get("metadata", {}),
            paid_at=data.get("paid_at"),
            created_at=data.get("created_at"),
            authorization=authorization,
            status=data.get("status"),
            gateway_response=data.get("gateway_response"),
            ip_address=data.get("ip_address"),
        )

        logger.info(
            "Parsed Paystack event '%s': ref=%s, amount=N%.2f, channel=%s",
            event.event,
            charge_data.reference,
            charge_data.amount,
            charge_data.channel,
        )
        return charge_data

    except KeyError as exc:
        logger.error("Missing required field in Paystack webhook data: %s", exc)
        raise ValueError(f"Missing required field in webhook data: {exc}") from exc
    except Exception as exc:
        logger.error("Failed to parse Paystack webhook payload: %s", exc)
        raise ValueError(f"Failed to parse webhook payload: {exc}") from exc
