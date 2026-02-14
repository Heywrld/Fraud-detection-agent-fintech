from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List
from enum import Enum


class TransactionChannel(str, Enum):
    POS = "pos"
    MOBILE_MONEY = "mobile_money"
    USSD = "ussd"
    BANK_TRANSFER = "bank_transfer"
    CARD = "card"


class TransactionStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    FLAGGED = "flagged"
    FROZEN = "frozen"


class CBNRiskLevel(str, Enum):
    """CBN risk classification tiers."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TransactionCreate(BaseModel):
    """Schema for creating a new transaction from webhook data."""
    paystack_ref: str
    amount_ngn: float = Field(gt=0, description="Transaction amount in Naira")
    channel: TransactionChannel
    customer_id: str
    customer_phone_hash: Optional[str] = None
    location_state: Optional[str] = None
    location_lga: Optional[str] = None
    device_fingerprint: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class Transaction(TransactionCreate):
    """Full transaction model with fraud scoring and CBN compliance."""
    id: str
    fraud_score: float = 0.0
    is_flagged: bool = False
    status: TransactionStatus = TransactionStatus.PENDING
    # CBN Compliance fields
    cbn_risk_level: Optional[CBNRiskLevel] = None
    cbn_risk_score: Optional[int] = None
    cbn_red_flags: Optional[List[str]] = Field(default_factory=list)
    file_str: bool = False
    cbn_recommendation: Optional[str] = None
    created_at: datetime


class TransactionResponse(BaseModel):
    """API response model for transactions with CBN compliance data."""
    id: str
    paystack_ref: str
    amount_ngn: float
    channel: TransactionChannel
    customer_id: str
    location_state: Optional[str] = None
    fraud_score: float
    is_flagged: bool
    status: TransactionStatus
    # CBN Compliance fields
    cbn_risk_level: Optional[CBNRiskLevel] = None
    cbn_risk_score: Optional[int] = None
    cbn_red_flags: Optional[List[str]] = Field(default_factory=list)
    file_str: bool = False
    cbn_recommendation: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class TransactionListResponse(BaseModel):
    """Paginated transaction list response."""
    transactions: list[TransactionResponse]
    total: int
    page: int
    page_size: int
