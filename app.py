import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error
from forecast import generate_forecast

st.set_page_config(page_title="Sales Horizon", layout="wide", page_icon="📈")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Sales Horizon")
    st.markdown("---")

    data_source = st.radio("Data Source", ["Sample Data", "Upload CSV"])
    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload sales CSV", type=["csv"])

    st.markdown("---")
    forecast_horizon = st.slider("Forecast Horizon (days)", min_value=7, max_value=90, value=30, step=7)

    method = st.selectbox(
        "Forecasting Method",
        options=["prophet", "exponential_smoothing", "moving_average", "naive"],
        format_func=lambda x: {
            "prophet": "Prophet (Facebook)",
            "exponential_smoothing": "Exponential Smoothing (Holt-Winters)",
            "moving_average": "Moving Average",
            "naive": "Naive Forecast",
        }[x],
    )

# ── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(source):
    if source is not None:
        return pd.read_csv(source)
    return pd.read_csv("data/sales.csv")

df = load_data(uploaded_file)
df["date"] = pd.to_datetime(df["date"])

# ── Filters ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.subheader("Filters")
    if "product_category" in df.columns:
        cats = ["All"] + sorted(df["product_category"].unique().tolist())
        selected_cat = st.selectbox("Product Category", cats)
    else:
        selected_cat = "All"

    if "region" in df.columns:
        regions = ["All"] + sorted(df["region"].unique().tolist())
        selected_region = st.selectbox("Region", regions)
    else:
        selected_region = "All"

filtered_df = df.copy()
if selected_cat != "All":
    filtered_df = filtered_df[filtered_df["product_category"] == selected_cat]
if selected_region != "All":
    filtered_df = filtered_df[filtered_df["region"] == selected_region]

daily_df = filtered_df.groupby("date", as_index=False)["sales"].sum()

# ── Forecast ─────────────────────────────────────────────────────────────────
@st.cache_data
def run_forecast(df_json, method, periods):
    df = pd.read_json(df_json)
    return generate_forecast(df, method=method, periods=periods)

with st.spinner("Generating forecast..."):
    forecast_df = run_forecast(daily_df.to_json(), method, forecast_horizon)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Forecast", "📈 Trends", "🎯 Accuracy", "📋 Data"])

# ── Tab 1: Forecast ───────────────────────────────────────────────────────────
with tab1:
    st.subheader("Sales Forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_df["date"], y=daily_df["sales"],
        name="Historical", line=dict(color="#636EFA")
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df["ds"], y=forecast_df["yhat"],
        name="Forecast", line=dict(color="#EF553B", dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df["ds"].tolist() + forecast_df["ds"].tolist()[::-1],
        y=forecast_df["yhat_upper"].tolist() + forecast_df["yhat_lower"].tolist()[::-1],
        fill="toself", fillcolor="rgba(239,85,59,0.1)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% Confidence Interval"
    ))
    fig.update_layout(hovermode="x unified", height=450)
    st.plotly_chart(fig, use_container_width=True)

    future_only = forecast_df[forecast_df["ds"] > daily_df["date"].max()]
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Forecast (daily)", f"${future_only[yhat].mean():,.0f}")
    col2.metric("Peak Forecast", f"${future_only[yhat].max():,.0f}")
    col3.metric("Forecast Period", f"{forecast_horizon} days")

    st.dataframe(future_only.rename(columns={"ds": "Date", "yhat": "Forecast", "yhat_lower": "Lower", "yhat_upper": "Upper"}).round(0), use_container_width=True)

# ── Tab 2: Trends ─────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Sales Trends")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Sales Distribution**")
        fig_hist = px.histogram(daily_df, x="sales", nbins=30, color_discrete_sequence=["#636EFA"])
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.markdown("**Day-of-Week Pattern**")
        daily_df["dow"] = daily_df["date"].dt.day_name()
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow_avg = daily_df.groupby("dow")["sales"].mean().reindex(dow_order)
        fig_dow = px.bar(dow_avg, color_discrete_sequence=["#EF553B"])
        st.plotly_chart(fig_dow, use_container_width=True)

    st.markdown("**Monthly Sales Trend**")
    monthly = daily_df.set_index("date")["sales"].resample("M").sum().reset_index()
    fig_monthly = px.line(monthly, x="date", y="sales", markers=True)
    st.plotly_chart(fig_monthly, use_container_width=True)

# ── Tab 3: Accuracy ───────────────────────────────────────────────────────────
with tab3:
    st.subheader("Forecast Accuracy")
    merged = daily_df.merge(forecast_df.rename(columns={"ds": "date"}), on="date", how="inner")
    if len(merged) > 10:
        mae = mean_absolute_error(merged["sales"], merged["yhat"])
        rmse = mean_squared_error(merged["sales"], merged["yhat"]) ** 0.5
        mape = (abs((merged["sales"] - merged["yhat"]) / merged["sales"].replace(0, np.nan)).mean()) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"${mae:,.0f}")
        col2.metric("RMSE", f"${rmse:,.0f}")
        col3.metric("MAPE", f"{mape:.1f}%")

        fig_scatter = px.scatter(merged, x="sales", y="yhat", trendline="ols",
                                  labels={"sales": "Actual", "yhat": "Predicted"},
                                  title="Actual vs Predicted")
        fig_scatter.add_shape(type="line", x0=merged["sales"].min(), y0=merged["sales"].min(),
                               x1=merged["sales"].max(), y1=merged["sales"].max(),
                               line=dict(dash="dash", color="red"))
        st.plotly_chart(fig_scatter, use_container_width=True)

        merged["residual"] = merged["sales"] - merged["yhat"]
        fig_res = px.line(merged, x="date", y="residual", title="Residuals Over Time")
        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_res, use_container_width=True)
    else:
        st.info("Not enough overlapping data to compute accuracy metrics. Upload more historical data.")

# ── Tab 4: Data ────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Historical Data")
    st.dataframe(daily_df.sort_values("date", ascending=False), use_container_width=True)
    csv_export = daily_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv_export, "sales_data.csv", "text/csv")
    st.markdown("**Summary Statistics**")
    st.dataframe(daily_df["sales"].describe().to_frame().T.round(2), use_container_width=True)
