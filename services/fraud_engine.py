"""
Core fraud detection orchestrator.

Coordinates the full lifecycle of fraud screening for every incoming
transaction:
    1. Persist raw transaction in Supabase
    2. Score with the ML model
    3. Flag / freeze based on thresholds
    4. Create alerts and log the audit trail
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from config import get_settings
from ml.predict import predict_fraud_score
from compliance.cbn import categorize_transaction_risk, evaluate_str_requirement
from services.transaction_service import get_customer_daily_stats

logger = logging.getLogger("fraud_guardian.engine")

# ---------------------------------------------------------------------------
# Internal helpers (Supabase interactions)
# ---------------------------------------------------------------------------


async def _store_transaction(tx: dict) -> dict:
    """Insert a raw transaction into the ``transactions`` table and return the row."""
    from db.supabase_client import get_supabase_client

    client = get_supabase_client()

    row = {
        "id": tx.get("id", str(uuid.uuid4())),
        "paystack_ref": tx.get("paystack_ref", tx.get("transaction_id", "")),
        "customer_id": tx["customer_id"],
        "amount_ngn": float(tx["amount_ngn"]),
        "channel": tx["channel"],
        "location_state": tx.get("location_state"),
        "location_lga": tx.get("location_lga"),
        "device_fingerprint": tx.get("device_fingerprint"),
        "metadata": tx.get("metadata", {}),
        "fraud_score": 0.0,
        "is_flagged": False,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        result = client.table("transactions").insert(row).execute()
        stored = result.data[0] if result.data else row
        logger.info(
            "Transaction %s stored for customer %s (%.2f NGN)",
            stored.get("id"), tx["customer_id"], float(tx["amount_ngn"]),
        )
        return stored
    except Exception as exc:
        logger.error("Failed to store transaction: %s", exc)
        # Return the local row so the pipeline can continue scoring even
        # if the DB write fails (resilience-first design).
        return row


async def _update_transaction(tx_id: str, updates: dict) -> None:
    """Patch an existing transaction row."""
    from db.supabase_client import get_supabase_client

    client = get_supabase_client()
    try:
        client.table("transactions").update(updates).eq("id", tx_id).execute()
    except Exception as exc:
        logger.error("Failed to update transaction %s: %s", tx_id, exc)


async def _create_alert(tx: dict, fraud_score: float) -> dict | None:
    """Create a fraud alert record and return it."""
    message = (
        f"Suspicious transaction detected: {float(tx['amount_ngn']):,.2f} NGN "
        f"via {tx.get('channel', 'unknown')} in {tx.get('location_state', 'unknown')}. "
        f"Fraud score: {fraud_score:.2f}."
    )
    return await _create_alert_with_message(tx, fraud_score, message)


async def _create_alert_with_message(tx: dict, fraud_score: float, message: str) -> dict | None:
    """Create a fraud alert record with a custom message."""
    from db.supabase_client import get_supabase_client

    client = get_supabase_client()

    alert = {
        "id": str(uuid.uuid4()),
        "transaction_id": tx["id"],
        "alert_type": "manual_review" if fraud_score < 0.9 else "freeze",
        "language": "en",
        "message": message,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        result = client.table("alerts").insert(alert).execute()
        stored_alert = result.data[0] if result.data else alert
        logger.info(
            "Alert %s created for transaction %s (score=%.2f)",
            stored_alert.get("id"), tx["id"], fraud_score,
        )
        return stored_alert
    except Exception as exc:
        logger.error("Failed to create alert for tx %s: %s", tx["id"], exc)
        return alert


async def _trigger_freeze(tx: dict) -> None:
    """
    Initiate an account freeze for the customer linked to the transaction.

    In a production system this would call an external freeze micro-service
    or banking API.  Here we log the action and update the DB.
    """
    from db.supabase_client import get_supabase_client

    customer_id = tx["customer_id"]
    logger.warning(
        "FREEZE triggered for customer %s due to transaction %s",
        customer_id, tx["id"],
    )

    client = get_supabase_client()
    try:
        freeze_record = {
            "id": str(uuid.uuid4()),
            "customer_id": customer_id,
            "transaction_id": tx["id"],
            "reason": f"Automated freeze -- fraud score exceeded threshold",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        client.table("account_freezes").insert(freeze_record).execute()
    except Exception as exc:
        # The freeze table may not exist yet; log but do not crash.
        logger.error("Failed to persist freeze record: %s", exc)


async def _log_audit(tx_id: str, actions: list[str], fraud_score: float) -> None:
    """Append an entry to the audit log."""
    from db.supabase_client import get_supabase_client

    client = get_supabase_client()
    entry = {
        "action": "transaction_processed",
        "entity_type": "transaction",
        "entity_id": tx_id,
        "details": {
            "fraud_score": fraud_score,
            "actions_taken": actions,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        client.table("audit_log").insert(entry).execute()
    except Exception as exc:
        logger.error("Failed to log audit trail for tx %s: %s", tx_id, exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def process_transaction(transaction_data: dict) -> dict:
    """
    End-to-end fraud screening pipeline for a single transaction.

    Parameters
    ----------
    transaction_data : dict
        Raw transaction payload.  Must include at minimum:
        customer_id, amount_ngn, channel.

    Returns
    -------
    dict
        The enriched transaction record with keys:
        - ``fraud_score`` (float 0-1)
        - ``is_flagged`` (bool)
        - ``status`` (str: pending | flagged | frozen)
        - ``actions`` (list[str]): human-readable list of actions taken
    """
    settings = get_settings()
    actions: list[str] = []

    # ---- 1. Store raw transaction in DB --------------------------------
    tx = await _store_transaction(transaction_data)
    tx_id = tx.get("id", transaction_data.get("id", "unknown"))
    actions.append("transaction_stored")

    # ---- 2. Score with ML model ----------------------------------------
    try:
        # Ensure required keys are present for the feature extractor
        scoring_payload = {
            "amount_ngn": float(tx.get("amount_ngn", transaction_data["amount_ngn"])),
            "channel": tx.get("channel", transaction_data.get("channel", "pos")),
            "location_state": tx.get("location_state", transaction_data.get("location_state", "Lagos")),
            "hour_of_day": tx.get("hour_of_day", transaction_data.get("hour_of_day", datetime.now().hour)),
            "day_of_week": tx.get("day_of_week", transaction_data.get("day_of_week", datetime.now().weekday())),
        }
        fraud_score = predict_fraud_score(scoring_payload)
        actions.append(f"ml_scored:{fraud_score:.4f}")
    except RuntimeError as exc:
        logger.error("ML model not available: %s", exc)
        fraud_score = 0.0
        actions.append("ml_score_failed:model_not_loaded")

    # ---- 2.5. CBN Compliance Evaluation ---------------------------------
    # Get customer daily stats for CBN threshold checks
    customer_id = tx.get("customer_id", transaction_data.get("customer_id"))
    daily_stats = await get_customer_daily_stats(customer_id)
    daily_total = daily_stats["daily_total"] + float(tx.get("amount_ngn", transaction_data["amount_ngn"]))
    daily_transaction_count = daily_stats["transaction_count"] + 1

    # Get transaction hour for CBN risk evaluation
    # Extract from created_at timestamp (stored in DB) or use current time
    tx_hour = datetime.now(timezone.utc).hour
    if tx.get("created_at"):
        try:
            if isinstance(tx["created_at"], str):
                tx_datetime = datetime.fromisoformat(tx["created_at"].replace("Z", "+00:00"))
            else:
                tx_datetime = tx["created_at"]
            if tx_datetime.tzinfo is None:
                tx_datetime = tx_datetime.replace(tzinfo=timezone.utc)
            tx_hour = tx_datetime.hour
        except Exception as exc:
            logger.warning("Failed to extract hour from created_at: %s", exc)
    elif transaction_data.get("hour_of_day") is not None:
        tx_hour = int(transaction_data["hour_of_day"])

    # Categorize transaction risk
    amount = float(tx.get("amount_ngn", transaction_data["amount_ngn"]))
    channel = tx.get("channel", transaction_data.get("channel", "pos"))
    
    risk_result = categorize_transaction_risk(
        amount=amount,
        channel=channel,
        hour_of_day=tx_hour,
        daily_total=daily_total,
        fraud_score=fraud_score,
    )
    actions.append(f"cbn_risk_categorized:{risk_result['risk_level'].value}")

    # Evaluate STR requirement
    # Note: customer_state would come from a customer profile lookup in production
    str_result = evaluate_str_requirement(
        amount=amount,
        channel=channel,
        hour_of_day=tx_hour,
        fraud_score=fraud_score,
        daily_total=daily_total,
        daily_transaction_count=daily_transaction_count,
        customer_state=transaction_data.get("customer_state"),  # Would come from customer profile
        transaction_state=tx.get("location_state", transaction_data.get("location_state")),
    )
    
    if str_result["file_str"]:
        actions.append("str_filing_required")
        logger.warning(
            "STR filing required for transaction %s: %d red flags detected",
            tx_id, len(str_result["red_flags"]),
        )

    # ---- 3. Update transaction with score and CBN data -----------------
    is_flagged = fraud_score > settings.fraud_score_flag_threshold
    status = "pending"

    if fraud_score > settings.fraud_score_freeze_threshold:
        status = "frozen"
        is_flagged = True
    elif is_flagged:
        status = "flagged"

    # Combine ML fraud score with CBN risk level for final decision
    # If CBN risk is CRITICAL or STR filing required, escalate the flag
    if str_result["file_str"] or risk_result["risk_level"].value == "critical":
        if not is_flagged:
            is_flagged = True
        if status == "pending":
            status = "flagged"
    
    update_fields: dict[str, Any] = {
        "fraud_score": fraud_score,
        "is_flagged": is_flagged,
        "status": status,
        # CBN compliance fields
        "cbn_risk_level": risk_result["risk_level"].value,
        "cbn_risk_score": risk_result["risk_score"],
        "cbn_red_flags": str_result["red_flags"],
        "file_str": str_result["file_str"],
        "cbn_recommendation": str_result["recommendation"],
    }
    await _update_transaction(tx_id, update_fields)
    actions.append(f"status_set:{status}")

    # ---- 4. Trigger freeze if needed -----------------------------------
    if status == "frozen":
        tx["id"] = tx_id  # ensure id is set
        await _trigger_freeze(tx)
        actions.append("account_freeze_triggered")

    # ---- 5. Create alert if flagged or STR required -------------------
    if is_flagged or str_result["file_str"]:
        tx["id"] = tx_id
        # Enhance alert message with CBN information if STR required
        if str_result["file_str"]:
            alert_message = (
                f"Suspicious transaction detected: {float(tx['amount_ngn']):,.2f} NGN "
                f"via {tx.get('channel', 'unknown')} in {tx.get('location_state', 'unknown')}. "
                f"Fraud score: {fraud_score:.2f}. "
                f"CBN Risk: {risk_result['risk_level'].value.upper()}. "
                f"STR filing required - {len(str_result['red_flags'])} red flags detected."
            )
        else:
            alert_message = (
                f"Suspicious transaction detected: {float(tx['amount_ngn']):,.2f} NGN "
                f"via {tx.get('channel', 'unknown')} in {tx.get('location_state', 'unknown')}. "
                f"Fraud score: {fraud_score:.2f}."
            )
        
        # Update alert creation to use enhanced message
        alert = await _create_alert_with_message(tx, fraud_score, alert_message)
        if alert:
            actions.append(f"alert_created:{alert.get('id', 'unknown')}")

    # ---- 6. Audit trail ------------------------------------------------
    await _log_audit(tx_id, actions, fraud_score)
    actions.append("audit_logged")

    # ---- 7. Build response ---------------------------------------------
    result = {
        "transaction_id": tx_id,
        "customer_id": tx.get("customer_id", transaction_data.get("customer_id")),
        "amount_ngn": float(tx.get("amount_ngn", transaction_data["amount_ngn"])),
        "channel": tx.get("channel", transaction_data.get("channel")),
        "location_state": tx.get("location_state", transaction_data.get("location_state")),
        "fraud_score": fraud_score,
        "is_flagged": is_flagged,
        "status": status,
        "actions": actions,
        # CBN compliance data
        "cbn_risk_level": risk_result["risk_level"].value,
        "cbn_risk_score": risk_result["risk_score"],
        "cbn_risk_factors": risk_result["factors"],
        "cbn_red_flags": str_result["red_flags"],
        "file_str": str_result["file_str"],
        "cbn_recommendation": str_result["recommendation"],
    }

    logger.info(
        "Transaction %s processed: score=%.4f status=%s cbn_risk=%s file_str=%s actions=%s",
        tx_id, fraud_score, status, risk_result["risk_level"].value, str_result["file_str"], actions,
    )

    return result
