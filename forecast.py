from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd
import numpy as np


def generate_forecast(df: pd.DataFrame, method: str = "prophet", periods: int = 90):
    """Generate a sales forecast using the specified method.

    Args:
        df: DataFrame with at least `date` and `sales` columns.
        method: One of "prophet", "exponential_smoothing", "moving_average", "naive".
        periods: Number of days to forecast.

    Returns:
        forecast_df: DataFrame with ds, yhat, yhat_lower, yhat_upper columns.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    if method == "prophet":
        return _prophet_forecast(df, periods)
    elif method == "exponential_smoothing":
        return _exponential_smoothing_forecast(df, periods)
    elif method == "moving_average":
        return _moving_average_forecast(df, periods)
    elif method == "naive":
        return _naive_forecast(df, periods)
    else:
        raise ValueError(f"Unknown forecasting method: {method}")


def _prophet_forecast(df: pd.DataFrame, periods: int) -> pd.DataFrame:
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    prophet_df = df.rename(columns={"date": "ds", "sales": "y"})
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def _exponential_smoothing_forecast(df: pd.DataFrame, periods: int) -> pd.DataFrame:
    series = df.set_index("date")["sales"].asfreq("D").fillna(method="ffill")
    model = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=7)
    fit = model.fit(optimized=True)
    forecast_vals = fit.forecast(periods)
    std = series.std()
    last_date = series.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
    hist = pd.DataFrame({
        "ds": series.index,
        "yhat": fit.fittedvalues.values,
        "yhat_lower": fit.fittedvalues.values - 1.96 * std,
        "yhat_upper": fit.fittedvalues.values + 1.96 * std,
    })
    fut = pd.DataFrame({
        "ds": future_dates,
        "yhat": forecast_vals.values,
        "yhat_lower": forecast_vals.values - 1.96 * std,
        "yhat_upper": forecast_vals.values + 1.96 * std,
    })
    return pd.concat([hist, fut], ignore_index=True)


def _moving_average_forecast(df: pd.DataFrame, periods: int, window: int = 14) -> pd.DataFrame:
    series = df.set_index("date")["sales"].asfreq("D").fillna(method="ffill")
    ma = series.rolling(window=window).mean()
    last_val = ma.dropna().iloc[-1]
    std = series.rolling(window=window).std().dropna().iloc[-1]
    last_date = series.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
    hist = pd.DataFrame({
        "ds": series.index,
        "yhat": ma.fillna(series).values,
        "yhat_lower": ma.fillna(series).values - 1.96 * std,
        "yhat_upper": ma.fillna(series).values + 1.96 * std,
    })
    fut = pd.DataFrame({
        "ds": future_dates,
        "yhat": [last_val] * periods,
        "yhat_lower": [last_val - 1.96 * std] * periods,
        "yhat_upper": [last_val + 1.96 * std] * periods,
    })
    return pd.concat([hist, fut], ignore_index=True)


def _naive_forecast(df: pd.DataFrame, periods: int) -> pd.DataFrame:
    series = df.set_index("date")["sales"].asfreq("D").fillna(method="ffill")
    last_val = series.iloc[-1]
    std = series.std()
    last_date = series.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
    hist = pd.DataFrame({
        "ds": series.index,
        "yhat": series.values,
        "yhat_lower": series.values - 1.96 * std,
        "yhat_upper": series.values + 1.96 * std,
    })
    fut = pd.DataFrame({
        "ds": future_dates,
        "yhat": [last_val] * periods,
        "yhat_lower": [last_val - 1.96 * std] * periods,
        "yhat_upper": [last_val + 1.96 * std] * periods,
    })
    return pd.concat([hist, fut], ignore_index=True)
