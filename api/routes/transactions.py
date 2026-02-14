"""
Transaction query endpoints.

Provides read-only access to transactions stored in Supabase,
including paginated listing with optional status filtering and
single-transaction lookup by ID.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status

from models.transaction import TransactionListResponse, TransactionResponse
from services.transaction_service import (
    get_transaction as svc_get_transaction,
    list_transactions as svc_list_transactions,
)

logger = logging.getLogger("fraud_guardian.routes.transactions")

router = APIRouter()


@router.get("/", response_model=TransactionListResponse)
async def list_transactions(
    status_filter: Optional[str] = Query(
        None,
        alias="status",
        description="Filter by transaction status (pending, approved, flagged, frozen)",
    ),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Records per page"),
) -> TransactionListResponse:
    """
    List transactions with optional filtering and pagination.

    Query parameters:
    - **status**: Filter by transaction status.
    - **page**: Page number (default 1).
    - **page_size**: Number of records per page (default 20, max 100).

    Returns a paginated list of transactions ordered by most recent first.
    """
    try:
        transactions, total = await svc_list_transactions(
            page=page,
            page_size=page_size,
            status=status_filter,
        )

        logger.info(
            "GET /transactions: status=%s page=%d page_size=%d returned=%d total=%d",
            status_filter, page, page_size, len(transactions), total,
        )

        return TransactionListResponse(
            transactions=[TransactionResponse(**tx) for tx in transactions],
            total=total,
            page=page,
            page_size=page_size,
        )

    except Exception as exc:
        logger.exception("Error listing transactions: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve transactions",
        )


@router.get("/{transaction_id}", response_model=TransactionResponse)
async def get_transaction(transaction_id: str) -> TransactionResponse:
    """
    Retrieve a single transaction by its UUID.

    Returns the full transaction record including fraud score and status,
    or 404 if the transaction does not exist.
    """
    try:
        tx = await svc_get_transaction(transaction_id)
    except Exception as exc:
        logger.exception("Error fetching transaction %s: %s", transaction_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve transaction",
        )

    if tx is None:
        logger.warning("Transaction not found: %s", transaction_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction {transaction_id} not found",
        )

    return TransactionResponse(**tx)
