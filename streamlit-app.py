import streamlit as st
import pandas as pd
from forecast import generate_forecast

st.set_page_config(page_title="Sales Horizon", layout="wide")

st.title("📈 Sales Horizon")
st.subheader("Intuitive Sales Forecast Dashboard")

uploaded_file = st.file_uploader("Upload sales CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("data/sales.csv")

st.write("### Raw Sales Data")
st.dataframe(df)

forecast_df, model = generate_forecast(df)

st.write("### Sales Forecast")
st.line_chart(forecast_df.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]])
