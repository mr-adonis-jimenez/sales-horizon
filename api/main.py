from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from api.models import (
    ForecastRequest, ForecastResponse, ForecastPoint,
    HealthResponse, ModelInfoResponse
)
from forecast import generate_forecast

app = FastAPI(
    title="Sales Horizon API",
    description="Production-ready sales forecasting REST API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_METHODS = ["prophet", "exponential_smoothing", "moving_average", "naive"]


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return HealthResponse(status="healthy", version="1.0.0")


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    return ModelInfoResponse(
        name="Sales Horizon Forecaster",
        version="1.0.0",
        supported_methods=SUPPORTED_METHODS,
        description="Time series sales forecasting with Prophet, Exponential Smoothing, Moving Average, and Naive methods.",
    )


@app.post("/predict", response_model=ForecastResponse, tags=["Forecast"])
def predict(request: ForecastRequest):
    try:
        df = pd.DataFrame([r.model_dump() for r in request.data])
        forecast_df = generate_forecast(df, method=request.method, periods=request.periods)
        # Return only future forecast points
        last_date = pd.to_datetime(df["date"]).max()
        future = forecast_df[forecast_df["ds"] > last_date]
        points = [
            ForecastPoint(
                ds=str(row["ds"])[:10],
                yhat=round(row["yhat"], 2),
                yhat_lower=round(row["yhat_lower"], 2),
                yhat_upper=round(row["yhat_upper"], 2),
            )
            for _, row in future.iterrows()
        ]
        return ForecastResponse(method=request.method, periods=request.periods, forecast=points)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
