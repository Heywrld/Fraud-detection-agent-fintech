"""
Account freeze and unfreeze service.

When fraud is detected with high confidence, customer accounts can be
frozen to prevent further suspicious transactions. All pending
transactions are held, an alert is generated, and an audit log entry
is written.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from db.supabase_client import get_supabase_client
from services.alert_service import create_alert

logger = logging.getLogger("fraud_guardian.services.freeze")


async def freeze_account(
    customer_id: str,
    transaction_id: str,
    reason: str,
) -> dict:
    """
    Freeze a customer account due to suspected fraud.

    Actions performed:
    1. Update all pending transactions for this customer to status='frozen'.
    2. Create a freeze alert record.
    3. Write an entry to the audit_log table.
    4. Return a summary freeze record.

    Args:
        customer_id: Hashed customer identifier.
        transaction_id: UUID of the transaction that triggered the freeze.
        reason: Human-readable reason for the freeze.

    Returns:
        Freeze record dict with customer_id, frozen_at, reason,
        frozen_transaction_count, and alert info.
    """
    client = get_supabase_client()
    frozen_at = datetime.now(timezone.utc).isoformat()

    # Step 1: Freeze all pending transactions for this customer
    frozen_count = 0
    try:
        response = (
            client.table("transactions")
            .update({"status": "frozen"})
            .eq("customer_id", customer_id)
            .eq("status", "pending")
            .execute()
        )
        frozen_count = len(response.data) if response.data else 0
        logger.info(
            "Froze %d pending transactions for customer %s",
            frozen_count,
            customer_id[:12] + "...",
        )
    except Exception as exc:
        logger.error(
            "Failed to freeze transactions for customer %s: %s",
            customer_id[:12] + "...",
            exc,
        )
        raise

    # Step 2: Create freeze alert
    alert = {}
    try:
        alert = await create_alert(
            transaction_id=transaction_id,
            alert_type="freeze",
            fraud_score=1.0,  # Freeze implies highest severity
        )
    except Exception as exc:
        logger.error("Failed to create freeze alert: %s", exc)
        # Continue even if alert creation fails -- the freeze itself is critical

    # Step 3: Write audit log entry
    try:
        audit_entry = {
            "action": "account_frozen",
            "entity_type": "customer",
            "entity_id": customer_id,
            "details": {
                "reason": reason,
                "transaction_id": transaction_id,
                "frozen_transaction_count": frozen_count,
            },
        }
        client.table("audit_log").insert(audit_entry).execute()
        logger.info("Audit log entry written for account freeze: customer %s", customer_id[:12] + "...")
    except Exception as exc:
        logger.error("Failed to write audit log for freeze: %s", exc)
        # Non-critical: log the error but do not fail the freeze operation

    # Step 4: Build and return freeze record
    freeze_record = {
        "customer_id": customer_id,
        "transaction_id": transaction_id,
        "frozen_at": frozen_at,
        "reason": reason,
        "frozen_transaction_count": frozen_count,
        "alert_id": alert.get("id"),
        "status": "frozen",
    }

    logger.info(
        "Account frozen: customer=%s, reason=%s, frozen_txns=%d",
        customer_id[:12] + "...",
        reason,
        frozen_count,
    )

    return freeze_record


async def unfreeze_account(
    customer_id: str,
    unfrozen_by: str,
) -> dict:
    """
    Unfreeze a previously frozen customer account.

    Actions performed:
    1. Update all frozen transactions for this customer to status='pending'
       (so they can be re-evaluated or manually approved).
    2. Write an audit log entry.
    3. Return a summary record.

    Args:
        customer_id: Hashed customer identifier.
        unfrozen_by: Name or ID of the person authorizing the unfreeze.

    Returns:
        Unfreeze record dict with customer_id, unfrozen_at, unfrozen_by,
        and unfrozen_transaction_count.
    """
    client = get_supabase_client()
    unfrozen_at = datetime.now(timezone.utc).isoformat()

    # Step 1: Restore frozen transactions to pending
    unfrozen_count = 0
    try:
        response = (
            client.table("transactions")
            .update({"status": "pending"})
            .eq("customer_id", customer_id)
            .eq("status", "frozen")
            .execute()
        )
        unfrozen_count = len(response.data) if response.data else 0
        logger.info(
            "Unfroze %d transactions for customer %s",
            unfrozen_count,
            customer_id[:12] + "...",
        )
    except Exception as exc:
        logger.error(
            "Failed to unfreeze transactions for customer %s: %s",
            customer_id[:12] + "...",
            exc,
        )
        raise

    # Step 2: Write audit log entry
    try:
        audit_entry = {
            "action": "account_unfrozen",
            "entity_type": "customer",
            "entity_id": customer_id,
            "details": {
                "unfrozen_by": unfrozen_by,
                "unfrozen_transaction_count": unfrozen_count,
            },
        }
        client.table("audit_log").insert(audit_entry).execute()
        logger.info(
            "Audit log entry written for account unfreeze: customer %s by %s",
            customer_id[:12] + "...",
            unfrozen_by,
        )
    except Exception as exc:
        logger.error("Failed to write audit log for unfreeze: %s", exc)

    # Step 3: Build and return unfreeze record
    unfreeze_record = {
        "customer_id": customer_id,
        "unfrozen_at": unfrozen_at,
        "unfrozen_by": unfrozen_by,
        "unfrozen_transaction_count": unfrozen_count,
        "status": "active",
    }

    logger.info(
        "Account unfrozen: customer=%s, by=%s, unfrozen_txns=%d",
        customer_id[:12] + "...",
        unfrozen_by,
        unfrozen_count,
    )

    return unfreeze_record
