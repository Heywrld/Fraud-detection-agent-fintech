"""
Dashboard endpoints for aggregated fraud statistics and recent activity.

Queries Supabase directly for aggregate counts (transactions, flags,
freezes, alert statuses) and returns pre-shaped payloads for the
front-end dashboard.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, status

from db.supabase_client import get_supabase_client
from models.transaction import TransactionResponse
from models.user import DashboardStats

logger = logging.getLogger("fraud_guardian.routes.dashboard")

router = APIRouter()


@router.get("/stats", response_model=DashboardStats)
async def dashboard_stats() -> DashboardStats:
    """
    Return aggregated dashboard statistics.

    Computes the following from the ``transactions`` and ``alerts`` tables:
    - total_transactions, total_flagged, total_frozen, flagged_percentage
    - alerts_pending, alerts_sent, alerts_resolved
    - flagged_by_channel and flagged_by_state breakdowns

    Fields that cannot be computed fall back to sensible defaults.
    """
    client = get_supabase_client()

    try:
        # ------------------------------------------------------------------
        # Transaction counts
        # ------------------------------------------------------------------
        total_resp = (
            client.table("transactions")
            .select("id", count="exact")
            .execute()
        )
        total_transactions = total_resp.count if total_resp.count is not None else 0

        flagged_resp = (
            client.table("transactions")
            .select("id", count="exact")
            .eq("is_flagged", True)
            .execute()
        )
        total_flagged = flagged_resp.count if flagged_resp.count is not None else 0

        frozen_resp = (
            client.table("transactions")
            .select("id", count="exact")
            .eq("status", "frozen")
            .execute()
        )
        total_frozen = frozen_resp.count if frozen_resp.count is not None else 0

        flagged_percentage = (
            round((total_flagged / total_transactions) * 100, 2)
            if total_transactions > 0
            else 0.0
        )

        # ------------------------------------------------------------------
        # Flagged breakdown by channel
        # ------------------------------------------------------------------
        flagged_by_channel: dict[str, int] = {}
        try:
            channel_resp = (
                client.table("transactions")
                .select("channel")
                .eq("is_flagged", True)
                .execute()
            )
            for row in (channel_resp.data or []):
                ch = row.get("channel", "unknown")
                flagged_by_channel[ch] = flagged_by_channel.get(ch, 0) + 1
        except Exception as exc:
            logger.warning("Could not compute flagged_by_channel: %s", exc)

        # ------------------------------------------------------------------
        # Flagged breakdown by state
        # ------------------------------------------------------------------
        flagged_by_state: dict[str, int] = {}
        try:
            state_resp = (
                client.table("transactions")
                .select("location_state")
                .eq("is_flagged", True)
                .execute()
            )
            for row in (state_resp.data or []):
                st = row.get("location_state") or "unknown"
                flagged_by_state[st] = flagged_by_state.get(st, 0) + 1
        except Exception as exc:
            logger.warning("Could not compute flagged_by_state: %s", exc)

        # ------------------------------------------------------------------
        # Alert counts by status
        # ------------------------------------------------------------------
        pending_resp = (
            client.table("alerts")
            .select("id", count="exact")
            .eq("status", "pending")
            .execute()
        )
        alerts_pending = pending_resp.count if pending_resp.count is not None else 0

        sent_resp = (
            client.table("alerts")
            .select("id", count="exact")
            .eq("status", "sent")
            .execute()
        )
        alerts_sent = sent_resp.count if sent_resp.count is not None else 0

        resolved_resp = (
            client.table("alerts")
            .select("id", count="exact")
            .eq("status", "resolved")
            .execute()
        )
        alerts_resolved = resolved_resp.count if resolved_resp.count is not None else 0

        # ------------------------------------------------------------------
        # Model version (best-effort)
        # ------------------------------------------------------------------
        model_version: Optional[str] = None
        try:
            mv_resp = (
                client.table("model_metrics")
                .select("model_version")
                .order("trained_at", desc=True)
                .limit(1)
                .execute()
            )
            if mv_resp.data:
                model_version = mv_resp.data[0].get("model_version")
        except Exception:
            # model_metrics table may not exist yet; that is fine
            pass

        logger.info(
            "GET /dashboard/stats: total=%d flagged=%d frozen=%d pending_alerts=%d",
            total_transactions, total_flagged, total_frozen, alerts_pending,
        )

        return DashboardStats(
            total_transactions=total_transactions,
            total_flagged=total_flagged,
            total_frozen=total_frozen,
            flagged_percentage=flagged_percentage,
            flagged_by_channel=flagged_by_channel,
            flagged_by_state=flagged_by_state,
            alerts_pending=alerts_pending,
            alerts_sent=alerts_sent,
            alerts_resolved=alerts_resolved,
            model_version=model_version,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error computing dashboard stats: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compute dashboard statistics",
        )


@router.get("/recent")
async def dashboard_recent() -> dict:
    """
    Return the 10 most recent flagged transactions.

    Queries the ``transactions`` table for rows where ``is_flagged=true``,
    ordered by ``created_at`` descending, limited to 10 records.
    """
    client = get_supabase_client()

    try:
        response = (
            client.table("transactions")
            .select("*")
            .eq("is_flagged", True)
            .order("created_at", desc=True)
            .limit(10)
            .execute()
        )

        transactions = response.data or []

        logger.info(
            "GET /dashboard/recent: returned=%d flagged transactions",
            len(transactions),
        )

        return {
            "transactions": [TransactionResponse(**tx) for tx in transactions],
        }

    except Exception as exc:
        logger.exception("Error fetching recent flagged transactions: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve recent flagged transactions",
        )
