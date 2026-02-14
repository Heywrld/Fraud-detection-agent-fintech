from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from enum import Enum


class AlertType(str, Enum):
    SMS = "sms"
    FREEZE = "freeze"
    MANUAL_REVIEW = "manual_review"


class AlertLanguage(str, Enum):
    ENGLISH = "en"
    HAUSA = "ha"
    YORUBA = "yo"
    PIDGIN = "pcm"


class AlertStatus(str, Enum):
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    RESOLVED = "resolved"


class AlertCreate(BaseModel):
    """Schema for creating a new alert."""
    transaction_id: str
    alert_type: AlertType
    language: AlertLanguage = AlertLanguage.ENGLISH
    message: str


class Alert(AlertCreate):
    """Full alert model."""
    id: str
    status: AlertStatus = AlertStatus.PENDING
    sent_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    created_at: datetime


class AlertResponse(BaseModel):
    """API response model for alerts."""
    id: str
    transaction_id: str
    alert_type: AlertType
    language: AlertLanguage
    message: str
    status: AlertStatus
    sent_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class AlertResolveRequest(BaseModel):
    """Request to resolve an alert."""
    resolved_by: str = Field(description="Name or ID of the person resolving the alert")
