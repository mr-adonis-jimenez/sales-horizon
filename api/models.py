from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from datetime import date


class SalesRecord(BaseModel):
    date: str = Field(..., example="2024-01-01")
    sales: float = Field(..., gt=0, example=15000.0)


class ForecastRequest(BaseModel):
    data: List[SalesRecord]
    method: Literal["prophet", "exponential_smoothing", "moving_average", "naive"] = "prophet"
    periods: int = Field(default=30, ge=7, le=90)


class ForecastPoint(BaseModel):
    ds: str
    yhat: float
    yhat_lower: float
    yhat_upper: float


class ForecastResponse(BaseModel):
    method: str
    periods: int
    forecast: List[ForecastPoint]


class HealthResponse(BaseModel):
    status: str
    version: str


class ModelInfoResponse(BaseModel):
    name: str
    version: str
    supported_methods: List[str]
    description: str
