from datetime import datetime

from pydantic import BaseModel, Field


class TransactionIn(BaseModel):
    transaction_id: str
    timestamp: datetime
    user_id: str
    amount: float = Field(gt=0)
    merchant_id: str
    category: str
    home_lat: float
    home_lon: float
    merchant_lat: float
    merchant_lon: float


class ScoreReason(BaseModel):
    feature: str
    value: float
    shap_contribution: float


class ScoreOut(BaseModel):
    transaction_id: str
    score: float
    decision: str
    threshold: float
    latency_ms: float
    top_reasons: list[ScoreReason]


class HealthOut(BaseModel):
    status: str
    model_version: str
    users_tracked: int
