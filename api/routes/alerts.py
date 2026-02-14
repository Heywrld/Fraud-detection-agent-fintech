"""
Alert management endpoints.

Provides paginated alert listing with optional status filtering and
an endpoint to resolve (acknowledge) an alert by its ID.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status

from models.alert import AlertResolveRequest, AlertResponse
from services.alert_service import (
    list_alerts as svc_list_alerts,
    resolve_alert as svc_resolve_alert,
)

logger = logging.getLogger("fraud_guardian.routes.alerts")

router = APIRouter()


@router.get("/")
async def list_alerts(
    status_filter: Optional[str] = Query(
        None,
        alias="status",
        description="Filter by alert status (pending, sent, failed, resolved)",
    ),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Records per page"),
) -> dict:
    """
    List alerts with optional status filtering and pagination.

    Query parameters:
    - **status**: Filter by alert status.
    - **page**: Page number (default 1).
    - **page_size**: Number of records per page (default 20, max 100).

    Returns a paginated list of alerts ordered by most recent first.
    """
    try:
        alerts, total = await svc_list_alerts(
            status=status_filter,
            page=page,
            page_size=page_size,
        )

        logger.info(
            "GET /alerts: status=%s page=%d page_size=%d returned=%d total=%d",
            status_filter, page, page_size, len(alerts), total,
        )

        return {
            "alerts": [AlertResponse(**a) for a in alerts],
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    except Exception as exc:
        logger.exception("Error listing alerts: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve alerts",
        )


@router.post("/{alert_id}/resolve", response_model=AlertResponse)
async def resolve_alert(
    alert_id: str,
    body: AlertResolveRequest,
) -> AlertResponse:
    """
    Mark an alert as resolved.

    Accepts the ID of the person or system resolving the alert in the
    request body.  Returns the updated alert record, or 404 if the
    alert does not exist.
    """
    try:
        alert = await svc_resolve_alert(
            alert_id=alert_id,
            resolved_by=body.resolved_by,
        )
    except Exception as exc:
        logger.exception("Error resolving alert %s: %s", alert_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to resolve alert",
        )

    if alert is None:
        logger.warning("Alert not found for resolution: %s", alert_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert {alert_id} not found",
        )

    logger.info("Alert %s resolved by %s", alert_id, body.resolved_by)
    return AlertResponse(**alert)
