from prophet import Prophet
import pandas as pd
import numpy as np


def generate_sample_data(num_days=365):
    """Generate sample sales data with trends, seasonality, and categories."""
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=num_days, freq="D")

    categories = ["Electronics", "Clothing", "Home & Garden", "Sports"]
    regions = ["North", "South", "East", "West"]

    rows = []
    for date in dates:
        for category in categories:
            for region in regions:
                base = {"Electronics": 15000, "Clothing": 8000, "Home & Garden": 6000, "Sports": 5000}[category]
                # weekly seasonality
                weekly = 1 + 0.15 * np.sin(2 * np.pi * date.dayofweek / 7)
                # monthly seasonality
                monthly = 1 + 0.1 * np.sin(2 * np.pi * date.month / 12)
                # upward trend
                trend = 1 + 0.0005 * (date - dates[0]).days
                # region factor
                region_factor = {"North": 1.0, "South": 0.9, "East": 0.85, "West": 1.1}[region]
                noise = np.random.normal(1.0, 0.08)
                sales = base * weekly * monthly * trend * region_factor * noise
                rows.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "sales": round(max(sales, 0), 2),
                    "product_category": category,
                    "region": region,
                })
    return pd.DataFrame(rows)


def forecast_prophet(df, periods=90):
    """Forecast using Facebook Prophet."""
    ts = df.groupby("date")["sales"].sum().reset_index()
    ts = ts.rename(columns={"date": "ds", "sales": "y"})
    ts["ds"] = pd.to_datetime(ts["ds"])

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(ts)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], ts


def forecast_exponential_smoothing(df, periods=90, alpha=0.3):
    """Forecast using Exponential Smoothing (Holt-Winters style)."""
    ts = df.groupby("date")["sales"].sum().reset_index()
    ts = ts.rename(columns={"date": "ds", "sales": "y"})
    ts["ds"] = pd.to_datetime(ts["ds"])
    ts = ts.sort_values("ds").reset_index(drop=True)

    values = ts["y"].values
    n = len(values)

    # Simple exponential smoothing with trend
    level = values[0]
    trend = (values[-1] - values[0]) / n if n > 1 else 0
    beta = 0.1

    fitted = []
    for i in range(n):
        if i == 0:
            fitted.append(level)
            continue
        prev_level = level
        level = alpha * values[i] + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
        fitted.append(level + trend)

    # Generate future predictions
    future_dates = pd.date_range(start=ts["ds"].iloc[-1] + pd.Timedelta(days=1), periods=periods, freq="D")
    predictions = []
    for i in range(1, periods + 1):
        predictions.append(level + trend * i)

    # Estimate confidence interval from residuals
    residuals = values - np.array(fitted)
    std_err = np.std(residuals)

    all_dates = pd.concat([ts["ds"], pd.Series(future_dates)], ignore_index=True)
    all_yhat = np.concatenate([fitted, predictions])
    steps = np.concatenate([np.ones(n), np.arange(1, periods + 1)])
    all_lower = all_yhat - 1.96 * std_err * np.sqrt(steps)
    all_upper = all_yhat + 1.96 * std_err * np.sqrt(steps)

    forecast_df = pd.DataFrame({
        "ds": all_dates,
        "yhat": all_yhat,
        "yhat_lower": all_lower,
        "yhat_upper": all_upper,
    })
    return forecast_df, ts


def forecast_moving_average(df, periods=90, window=30):
    """Forecast using Moving Average."""
    ts = df.groupby("date")["sales"].sum().reset_index()
    ts = ts.rename(columns={"date": "ds", "sales": "y"})
    ts["ds"] = pd.to_datetime(ts["ds"])
    ts = ts.sort_values("ds").reset_index(drop=True)

    values = ts["y"].values
    n = len(values)

    # Calculate moving average for historical data
    fitted = []
    for i in range(n):
        start = max(0, i - window + 1)
        fitted.append(np.mean(values[start:i + 1]))

    # Forecast: use last window values
    last_window = values[-window:] if n >= window else values
    ma_value = np.mean(last_window)

    future_dates = pd.date_range(start=ts["ds"].iloc[-1] + pd.Timedelta(days=1), periods=periods, freq="D")
    predictions = np.full(periods, ma_value)

    std_err = np.std(values[-window:]) if n >= window else np.std(values)
    steps = np.arange(1, periods + 1)

    all_dates = pd.concat([ts["ds"], pd.Series(future_dates)], ignore_index=True)
    all_yhat = np.concatenate([fitted, predictions])
    all_lower = np.concatenate([
        np.array(fitted) - 1.96 * std_err,
        predictions - 1.96 * std_err * np.sqrt(steps / window),
    ])
    all_upper = np.concatenate([
        np.array(fitted) + 1.96 * std_err,
        predictions + 1.96 * std_err * np.sqrt(steps / window),
    ])

    forecast_df = pd.DataFrame({
        "ds": all_dates,
        "yhat": all_yhat,
        "yhat_lower": all_lower,
        "yhat_upper": all_upper,
    })
    return forecast_df, ts


def forecast_naive(df, periods=90):
    """Forecast using Naive method (last value carried forward)."""
    ts = df.groupby("date")["sales"].sum().reset_index()
    ts = ts.rename(columns={"date": "ds", "sales": "y"})
    ts["ds"] = pd.to_datetime(ts["ds"])
    ts = ts.sort_values("ds").reset_index(drop=True)

    values = ts["y"].values
    n = len(values)
    last_value = values[-1]

    future_dates = pd.date_range(start=ts["ds"].iloc[-1] + pd.Timedelta(days=1), periods=periods, freq="D")
    predictions = np.full(periods, last_value)

    std_err = np.std(np.diff(values)) if n > 1 else 0
    steps = np.arange(1, periods + 1)

    all_dates = pd.concat([ts["ds"], pd.Series(future_dates)], ignore_index=True)
    all_yhat = np.concatenate([values, predictions])
    all_lower = np.concatenate([
        values - 1.96 * std_err,
        predictions - 1.96 * std_err * np.sqrt(steps),
    ])
    all_upper = np.concatenate([
        values + 1.96 * std_err,
        predictions + 1.96 * std_err * np.sqrt(steps),
    ])

    forecast_df = pd.DataFrame({
        "ds": all_dates,
        "yhat": all_yhat,
        "yhat_lower": all_lower,
        "yhat_upper": all_upper,
    })
    return forecast_df, ts


def generate_forecast(df, method="Prophet", periods=90):
    """Dispatch to the selected forecasting method."""
    methods = {
        "Prophet": forecast_prophet,
        "Exponential Smoothing": forecast_exponential_smoothing,
        "Moving Average": forecast_moving_average,
        "Naive": forecast_naive,
    }
    fn = methods.get(method, forecast_prophet)
    return fn(df, periods=periods)
