from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class AuditLogEntry(BaseModel):
    """Audit trail entry."""
    id: str
    action: str
    entity_type: str
    entity_id: Optional[str] = None
    details: dict = {}
    ip_address: Optional[str] = None
    created_at: datetime


class ModelMetrics(BaseModel):
    """ML model performance metrics."""
    id: str
    model_version: str
    accuracy: Optional[float] = None
    precision_score: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    bias_report: dict = {}
    trained_at: datetime


class DashboardStats(BaseModel):
    """Dashboard statistics response."""
    total_transactions: int
    total_flagged: int
    total_frozen: int
    flagged_percentage: float
    flagged_by_channel: dict
    flagged_by_state: dict
    alerts_pending: int
    alerts_sent: int
    alerts_resolved: int
    model_version: Optional[str] = None
