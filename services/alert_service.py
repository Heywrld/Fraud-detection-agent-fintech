"""
Alert creation, SMS dispatch, and lifecycle management service.

Alerts are stored in the Supabase 'alerts' table and optionally
delivered via SMS through the Twilio integration.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from db.supabase_client import get_supabase_client
from integrations.twilio.client import get_twilio_client
from integrations.twilio.templates import get_language_for_state, get_template

logger = logging.getLogger("fraud_guardian.services.alert")


async def create_alert(
    transaction_id: str,
    alert_type: str,
    fraud_score: float,
    customer_state: Optional[str] = None,
    customer_phone: Optional[str] = None,
) -> dict:
    """
    Create a fraud alert, optionally sending an SMS notification.

    Workflow:
    1. Determine the customer's likely language from their state.
    2. Select the appropriate template type based on fraud_score:
       - score >= 0.9 -> account_frozen template
       - score >= 0.7 -> fraud_alert template
       - else          -> transaction_flagged template
    3. Render the message in the chosen language.
    4. Store the alert record in Supabase.
    5. If alert_type is 'sms' and a phone number is provided, send the SMS.
    6. Update the alert status to 'sent' or 'failed'.

    Args:
        transaction_id: UUID of the flagged transaction.
        alert_type: Type of alert ('sms', 'freeze', 'manual_review').
        fraud_score: The computed fraud score (0.0 - 1.0).
        customer_state: Nigerian state name for language selection.
        customer_phone: Customer phone number for SMS delivery.

    Returns:
        The alert record dict as stored in Supabase.
    """
    client = get_supabase_client()

    # Step 1: Determine language
    language = get_language_for_state(customer_state)

    # Step 2: Pick template based on fraud severity
    if fraud_score >= 0.9:
        template_type = "account_frozen"
    elif fraud_score >= 0.7:
        template_type = "fraud_alert"
    else:
        template_type = "transaction_flagged"

    # Step 3: Render message -- fetch the transaction for template variables
    template_kwargs = {}
    try:
        txn_response = (
            client.table("transactions")
            .select("amount_ngn, channel")
            .eq("id", transaction_id)
            .execute()
        )
        if txn_response.data:
            txn = txn_response.data[0]
            template_kwargs["amount"] = f"{txn.get('amount_ngn', 0):,.2f}"
            template_kwargs["channel"] = txn.get("channel", "unknown")
    except Exception as exc:
        logger.warning("Could not fetch transaction %s for template: %s", transaction_id, exc)
        template_kwargs["amount"] = "0.00"
        template_kwargs["channel"] = "unknown"

    message = get_template(template_type, language, **template_kwargs)

    # Step 4: Insert alert record
    alert_record = {
        "transaction_id": transaction_id,
        "alert_type": alert_type,
        "language": language,
        "message": message,
        "status": "pending",
    }

    try:
        response = client.table("alerts").insert(alert_record).execute()
        alert = response.data[0] if response.data else alert_record
    except Exception as exc:
        logger.error("Failed to insert alert for transaction %s: %s", transaction_id, exc)
        raise

    alert_id = alert.get("id")
    logger.info(
        "Created alert %s: type=%s, language=%s, template=%s",
        alert_id, alert_type, language, template_type,
    )

    # Step 5: Send SMS if applicable
    if alert_type == "sms" and customer_phone:
        try:
            twilio_client = get_twilio_client()
            sms_result = twilio_client.send_sms(to=customer_phone, message=message)

            if sms_result["status"] in ("sent", "logged"):
                new_status = "sent"
            else:
                new_status = "failed"

            logger.info(
                "SMS dispatch for alert %s: status=%s, sid=%s",
                alert_id,
                sms_result["status"],
                sms_result.get("sid"),
            )
        except Exception as exc:
            logger.error("SMS dispatch failed for alert %s: %s", alert_id, exc)
            new_status = "failed"

        # Step 6: Update alert status
        try:
            update_data = {"status": new_status}
            if new_status == "sent":
                update_data["sent_at"] = datetime.now(timezone.utc).isoformat()

            update_response = (
                client.table("alerts")
                .update(update_data)
                .eq("id", alert_id)
                .execute()
            )
            if update_response.data:
                alert = update_response.data[0]
        except Exception as exc:
            logger.error("Failed to update alert %s status: %s", alert_id, exc)

    return alert


async def resolve_alert(alert_id: str, resolved_by: str) -> Optional[dict]:
    """
    Mark an alert as resolved.

    Args:
        alert_id: UUID of the alert to resolve.
        resolved_by: Name or ID of the person resolving the alert.

    Returns:
        Updated alert record, or None if the alert was not found.
    """
    client = get_supabase_client()

    try:
        response = (
            client.table("alerts")
            .update({
                "status": "resolved",
                "resolved_at": datetime.now(timezone.utc).isoformat(),
                "resolved_by": resolved_by,
            })
            .eq("id", alert_id)
            .execute()
        )

        if response.data:
            logger.info("Alert %s resolved by %s", alert_id, resolved_by)
            return response.data[0]

        logger.warning("Alert not found for resolution: %s", alert_id)
        return None

    except Exception as exc:
        logger.error("Failed to resolve alert %s: %s", alert_id, exc)
        raise


async def list_alerts(
    status: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list, int]:
    """
    List alerts with optional status filtering and pagination.

    Args:
        status: Filter by alert status ('pending', 'sent', 'failed', 'resolved').
        page: Page number (1-indexed).
        page_size: Number of records per page.

    Returns:
        Tuple of (list of alert dicts, total count).
    """
    client = get_supabase_client()

    try:
        query = client.table("alerts").select("*", count="exact")

        if status is not None:
            query = query.eq("status", status)

        start = (page - 1) * page_size
        end = start + page_size - 1

        response = (
            query
            .order("created_at", desc=True)
            .range(start, end)
            .execute()
        )

        alerts = response.data or []
        total = response.count if response.count is not None else len(alerts)

        logger.info(
            "Listed alerts: page=%d, page_size=%d, returned=%d, total=%d",
            page, page_size, len(alerts), total,
        )
        return alerts, total

    except Exception as exc:
        logger.error("Failed to list alerts: %s", exc)
        raise
