import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
from forecast import generate_forecast, generate_sample_data

st.set_page_config(page_title="Sales Horizon", layout="wide")

st.title("Sales Horizon")
st.subheader("Intuitive Sales Forecast Dashboard")

# ── Sidebar Controls ──────────────────────────────────────────────────────────

st.sidebar.header("Controls")

data_source = st.sidebar.radio("Data Source", ["Sample Data", "Upload CSV"])

forecast_horizon = st.sidebar.slider("Forecast Horizon (days)", min_value=7, max_value=90, value=30, step=7)

forecast_method = st.sidebar.selectbox(
    "Forecasting Method",
    ["Exponential Smoothing", "Moving Average", "Naive", "Prophet"],
    help=(
        "Exponential Smoothing: Best for data with trend and seasonality. "
        "Moving Average: Simple method for stable trends. "
        "Naive: Uses last known value as prediction. "
        "Prophet: Facebook's advanced time series model."
    ),
)

# ── Data Loading ──────────────────────────────────────────────────────────────

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload sales CSV", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if "date" not in df.columns or "sales" not in df.columns:
                st.error("CSV must contain 'date' and 'sales' columns.")
                st.stop()
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.stop()
    else:
        st.info("Please upload a CSV file, or switch to Sample Data.")
        st.stop()
else:
    df = generate_sample_data()

# ── Sidebar Filters ───────────────────────────────────────────────────────────

if "product_category" in df.columns:
    categories = sorted(df["product_category"].unique())
    selected_categories = st.sidebar.multiselect("Product Category", categories, default=categories)
    df = df[df["product_category"].isin(selected_categories)]

if "region" in df.columns:
    regions = sorted(df["region"].unique())
    selected_regions = st.sidebar.multiselect("Region", regions, default=regions)
    df = df[df["region"].isin(selected_regions)]

if df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# ── Generate Forecast ─────────────────────────────────────────────────────────

try:
    with st.spinner(f"Running {forecast_method} forecast..."):
        forecast_df, historical_ts = generate_forecast(df, method=forecast_method, periods=forecast_horizon)
except Exception as e:
    st.error(f"Forecasting failed: {e}")
    st.stop()

# Split into historical fitted and future predictions
n_hist = len(historical_ts)
hist_forecast = forecast_df.iloc[:n_hist].copy()
future_forecast = forecast_df.iloc[n_hist:].copy()

# ── Dashboard Tabs ────────────────────────────────────────────────────────────

tab_forecast, tab_trends, tab_accuracy, tab_data = st.tabs(
    ["Forecast", "Trends", "Accuracy", "Data"]
)

# ── Forecast Tab ──────────────────────────────────────────────────────────────

with tab_forecast:
    st.markdown("### Sales Forecast")

    fig = go.Figure()

    # Historical actual
    fig.add_trace(go.Scatter(
        x=historical_ts["ds"], y=historical_ts["y"],
        mode="lines", name="Actual Sales",
        line=dict(color="#636EFA"),
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=future_forecast["ds"], y=future_forecast["yhat"],
        mode="lines", name="Forecast",
        line=dict(color="#EF553B", dash="dash"),
    ))

    # Confidence interval
    fig.add_trace(go.Scatter(
        x=pd.concat([future_forecast["ds"], future_forecast["ds"][::-1]]),
        y=pd.concat([future_forecast["yhat_upper"], future_forecast["yhat_lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(239,85,59,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% Confidence",
    ))

    fig.update_layout(
        xaxis_title="Date", yaxis_title="Sales",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Last Actual", f"${historical_ts['y'].iloc[-1]:,.0f}")
    col2.metric("Avg Forecast", f"${future_forecast['yhat'].mean():,.0f}")
    col3.metric(
        "Forecast Trend",
        f"{'Up' if future_forecast['yhat'].iloc[-1] > historical_ts['y'].iloc[-1] else 'Down'}",
        delta=f"{((future_forecast['yhat'].iloc[-1] / historical_ts['y'].iloc[-1]) - 1) * 100:+.1f}%",
    )
    col4.metric("Forecast Period", f"{forecast_horizon} days")

    st.markdown("### Forecast Summary")
    summary = future_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    summary.columns = ["Date", "Forecast", "Lower Bound", "Upper Bound"]
    summary["Forecast"] = summary["Forecast"].round(2)
    summary["Lower Bound"] = summary["Lower Bound"].round(2)
    summary["Upper Bound"] = summary["Upper Bound"].round(2)
    st.dataframe(summary, use_container_width=True, hide_index=True)

# ── Trends Tab ────────────────────────────────────────────────────────────────

with tab_trends:
    st.markdown("### Sales Distribution")
    fig_hist = px.histogram(
        historical_ts, x="y", nbins=40,
        labels={"y": "Daily Sales"},
        color_discrete_sequence=["#636EFA"],
    )
    fig_hist.update_layout(height=350)
    st.plotly_chart(fig_hist, use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Day-of-Week Analysis")
        ts_dow = historical_ts.copy()
        ts_dow["day"] = ts_dow["ds"].dt.day_name()
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_avg = ts_dow.groupby("day")["y"].mean().reindex(dow_order)
        fig_dow = px.bar(
            x=dow_order, y=dow_avg.values,
            labels={"x": "Day of Week", "y": "Avg Sales"},
            color_discrete_sequence=["#00CC96"],
        )
        fig_dow.update_layout(height=350)
        st.plotly_chart(fig_dow, use_container_width=True)

    with col_right:
        st.markdown("### Monthly Trend")
        ts_month = historical_ts.copy()
        ts_month["month"] = ts_month["ds"].dt.to_period("M").astype(str)
        monthly_sum = ts_month.groupby("month")["y"].sum().reset_index()
        fig_month = px.bar(
            monthly_sum, x="month", y="y",
            labels={"month": "Month", "y": "Total Sales"},
            color_discrete_sequence=["#AB63FA"],
        )
        fig_month.update_layout(height=350)
        st.plotly_chart(fig_month, use_container_width=True)

# ── Accuracy Tab ──────────────────────────────────────────────────────────────

with tab_accuracy:
    st.markdown("### Forecast Accuracy Metrics")
    st.caption("Evaluated on fitted values vs. actuals over the historical period.")

    actual = historical_ts["y"].values
    fitted = hist_forecast["yhat"].values

    mae = mean_absolute_error(actual, fitted)
    rmse = np.sqrt(mean_squared_error(actual, fitted))
    mape = np.mean(np.abs((actual - fitted) / actual)) * 100

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"${mae:,.2f}")
    m2.metric("RMSE", f"${rmse:,.2f}")
    m3.metric("MAPE", f"{mape:.2f}%")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Actual vs Predicted")
        fig_scatter = px.scatter(
            x=actual, y=fitted,
            labels={"x": "Actual", "y": "Predicted"},
            color_discrete_sequence=["#636EFA"],
        )
        max_val = max(actual.max(), fitted.max())
        min_val = min(actual.min(), fitted.min())
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines", name="Perfect Fit",
            line=dict(color="gray", dash="dash"),
        ))
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_b:
        st.markdown("### Residuals Over Time")
        residuals = actual - fitted
        fig_resid = px.bar(
            x=historical_ts["ds"], y=residuals,
            labels={"x": "Date", "y": "Residual (Actual - Predicted)"},
            color_discrete_sequence=["#EF553B"],
        )
        fig_resid.update_layout(height=400)
        st.plotly_chart(fig_resid, use_container_width=True)

# ── Data Tab ──────────────────────────────────────────────────────────────────

with tab_data:
    st.markdown("### Historical Data")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### Data Statistics")
    daily_totals = df.groupby("date")["sales"].sum()
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Records", f"{len(df):,}")
    s2.metric("Date Range", f"{df['date'].min()} to {df['date'].max()}")
    s3.metric("Avg Daily Sales", f"${daily_totals.mean():,.0f}")
    s4.metric("Max Daily Sales", f"${daily_totals.max():,.0f}")

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Data as CSV",
        data=csv_bytes,
        file_name="sales_horizon_data.csv",
        mime="text/csv",
    )
