"""
Transaction CRUD service backed by Supabase.

Provides async functions for creating, reading, updating, and listing
transactions, as well as velocity-check queries for fraud detection.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from db.supabase_client import get_supabase_client

logger = logging.getLogger("fraud_guardian.services.transaction")


async def create_transaction(data: dict) -> dict:
    """
    Insert a new transaction record into the transactions table.

    Args:
        data: Dict matching the TransactionCreate schema fields.
              Expected keys: paystack_ref, amount_ngn, channel,
              customer_id, customer_phone_hash, location_state,
              device_fingerprint, metadata.

    Returns:
        The inserted transaction record as a dict (including generated id).

    Raises:
        Exception: On Supabase insert failure.
    """
    client = get_supabase_client()

    # Set default values for scoring fields
    record = {
        **data,
        "fraud_score": data.get("fraud_score", 0.0),
        "is_flagged": data.get("is_flagged", False),
        "status": data.get("status", "pending"),
    }

    try:
        response = client.table("transactions").insert(record).execute()
        inserted = response.data[0] if response.data else {}
        logger.info(
            "Created transaction: id=%s, ref=%s, amount=N%.2f",
            inserted.get("id"),
            inserted.get("paystack_ref"),
            inserted.get("amount_ngn", 0),
        )
        return inserted
    except Exception as exc:
        logger.error("Failed to create transaction: %s", exc)
        raise


async def get_transaction(transaction_id: str) -> Optional[dict]:
    """
    Retrieve a single transaction by its UUID.

    Args:
        transaction_id: The transaction UUID.

    Returns:
        Transaction record dict, or None if not found.
    """
    client = get_supabase_client()

    try:
        response = (
            client.table("transactions")
            .select("*")
            .eq("id", transaction_id)
            .execute()
        )
        if response.data:
            return response.data[0]
        logger.warning("Transaction not found: %s", transaction_id)
        return None
    except Exception as exc:
        logger.error("Failed to get transaction %s: %s", transaction_id, exc)
        raise


async def update_transaction(transaction_id: str, updates: dict) -> Optional[dict]:
    """
    Update a transaction record by its UUID.

    Args:
        transaction_id: The transaction UUID to update.
        updates: Dict of fields to update (e.g. fraud_score, is_flagged, status).

    Returns:
        The updated transaction record, or None if not found.
    """
    client = get_supabase_client()

    try:
        response = (
            client.table("transactions")
            .update(updates)
            .eq("id", transaction_id)
            .execute()
        )
        if response.data:
            updated = response.data[0]
            logger.info("Updated transaction %s: %s", transaction_id, list(updates.keys()))
            return updated
        logger.warning("Transaction not found for update: %s", transaction_id)
        return None
    except Exception as exc:
        logger.error("Failed to update transaction %s: %s", transaction_id, exc)
        raise


async def list_transactions(
    page: int = 1,
    page_size: int = 20,
    status: Optional[str] = None,
    is_flagged: Optional[bool] = None,
) -> tuple[list, int]:
    """
    List transactions with optional filtering and pagination.

    Args:
        page: Page number (1-indexed).
        page_size: Number of records per page.
        status: Filter by transaction status (pending, approved, flagged, frozen).
        is_flagged: Filter by whether the transaction is flagged.

    Returns:
        Tuple of (list of transaction dicts, total count).
    """
    client = get_supabase_client()

    try:
        # Build the query
        query = client.table("transactions").select("*", count="exact")

        if status is not None:
            query = query.eq("status", status)
        if is_flagged is not None:
            query = query.eq("is_flagged", is_flagged)

        # Pagination: Supabase range is 0-indexed and inclusive
        start = (page - 1) * page_size
        end = start + page_size - 1

        response = (
            query
            .order("created_at", desc=True)
            .range(start, end)
            .execute()
        )

        transactions = response.data or []
        total = response.count if response.count is not None else len(transactions)

        logger.info(
            "Listed transactions: page=%d, page_size=%d, returned=%d, total=%d",
            page, page_size, len(transactions), total,
        )
        return transactions, total

    except Exception as exc:
        logger.error("Failed to list transactions: %s", exc)
        raise


async def get_customer_recent_transactions(
    customer_id: str,
    hours: int = 24,
) -> list:
    """
    Retrieve recent transactions for a customer within a time window.

    Used for velocity checks during fraud scoring -- e.g. detecting
    unusually high transaction frequency.

    Args:
        customer_id: Hashed customer identifier.
        hours: Number of hours to look back (default 24).

    Returns:
        List of transaction dicts within the time window, ordered by
        most recent first.
    """
    client = get_supabase_client()

    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

    try:
        response = (
            client.table("transactions")
            .select("*")
            .eq("customer_id", customer_id)
            .gte("created_at", cutoff)
            .order("created_at", desc=True)
            .execute()
        )

        transactions = response.data or []
        logger.info(
            "Customer %s has %d transactions in the last %d hours",
            customer_id[:12] + "...",
            len(transactions),
            hours,
        )
        return transactions

    except Exception as exc:
        logger.error(
            "Failed to get recent transactions for customer %s: %s",
            customer_id[:12] + "...",
            exc,
        )
        raise


async def get_customer_daily_stats(
    customer_id: str,
    date: Optional[datetime] = None,
) -> dict:
    """
    Calculate daily transaction statistics for a customer on a specific date.

    Used for CBN compliance checks -- calculates cumulative daily amounts
    and transaction counts for threshold evaluation.

    Args:
        customer_id: Hashed customer identifier.
        date: Date to calculate stats for (defaults to today in UTC).
              If provided, should be timezone-aware or will be treated as UTC.

    Returns:
        Dict with keys:
        - ``daily_total``: Sum of all transaction amounts for the day (float)
        - ``transaction_count``: Number of transactions on the day (int)
    """
    client = get_supabase_client()

    if date is None:
        date = datetime.now(timezone.utc)
    elif date.tzinfo is None:
        date = date.replace(tzinfo=timezone.utc)

    # Get start and end of day in UTC
    start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = start_of_day.replace(hour=23, minute=59, second=59, microsecond=999999)

    try:
        response = (
            client.table("transactions")
            .select("amount_ngn")
            .eq("customer_id", customer_id)
            .gte("created_at", start_of_day.isoformat())
            .lte("created_at", end_of_day.isoformat())
            .execute()
        )

        transactions = response.data or []
        daily_total = sum(float(tx.get("amount_ngn", 0)) for tx in transactions)
        transaction_count = len(transactions)

        logger.info(
            "Customer %s daily stats: total=N%.2f, count=%d",
            customer_id[:12] + "...",
            daily_total,
            transaction_count,
        )

        return {
            "daily_total": daily_total,
            "transaction_count": transaction_count,
        }

    except Exception as exc:
        logger.error(
            "Failed to get daily stats for customer %s: %s",
            customer_id[:12] + "...",
            exc,
        )
        # Return safe defaults on error
        return {
            "daily_total": 0.0,
            "transaction_count": 0,
        }
