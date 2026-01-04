import streamlit as st
import pandas as pd
from forecast import generate_forecast

if uploaded_file:
    df = pd.read_csv("data/sales.csv")
else:
    df = pd.read_csv("data/sales.csv")

st.write("### Raw Sales Data")
st.dataframe(df)

forecast_df, model = generate_forecast(df)

st.write("### Sales Forecast")
st.line_chart(forecast_df.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]])
