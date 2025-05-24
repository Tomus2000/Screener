# -*- coding: utf-8 -*-
"""Growth Stock Screener Dashboard using Finnhub and YFinance"""

import requests
import time

# 🔑 Finnhub API key
FINNHUB_API_KEY = "d0hiea9r01qup0c6eeugd0hiea9r01qup0c6eev0"

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

st.set_page_config(layout="wide")

st.title("📊 Growth Stock Screener Dashboard")
st.markdown("Analyze growth, quality, momentum, valuation, and dividend metrics across your custom watchlist.")

# Sidebar input
st.sidebar.header("⚙️ Stock Selection")
tickers_input = st.sidebar.text_input(
    "Enter tickers (comma-separated)", 
    value="AXON, CELH, DUOL, INTA, IOT, APP, ENPH, ON, DT, GLOB, ADYEN"
)

# Filters
st.sidebar.subheader("🔍 Filters")
min_score = st.sidebar.slider("Minimum Investment Score", 1, 100, 1)
min_yield = st.sidebar.slider("Minimum Dividend Yield (%)", 0.0, 10.0, 0.0)

# Convert user input into a list
watchlist = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

price_data = {}
results = []

# Finnhub data fetcher
finnhub_url = "https://finnhub.io/api/v1"
def get_finnhub_json(endpoint, params):
    params['token'] = FINNHUB_API_KEY
    response = requests.get(f"{finnhub_url}/{endpoint}", params=params)
    return response.json() if response.status_code == 200 else {}

# Display dataframe
df = pd.DataFrame(results).fillna(0)
st.subheader("📋 Screener Table")
st.dataframe(df.set_index("Ticker"))

st.markdown("""
**📘 Investment Score Explained:**
- **Growth (35%)**: Revenue and EPS growth (YoY).
- **Momentum (20%)**: 12-month price performance.
- **Quality (20%)**: Return on equity and profit margin.
- **Earnings Momentum (15%)**: Latest earnings surprise (% vs. estimate).
- **Valuation (10%)**: Moderate P/E rewarded (below 50).
Scores are normalized and scaled from 1 to 100.
""")

# Excel export
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")

csv = convert_df(df)
st.download_button(
    label="⬇️ Download Screener Results as CSV",
    data=csv,
    file_name="growth_screener_results.csv",
    mime="text/csv"
)

# Heatmap
st.subheader("🔥 Interactive Heatmap of Key Metrics")
heatmap_df = df.set_index("Ticker")[[
    "Rev Growth", "EPS Growth", "Earnings Surprise (%)",
    "ROE", "Profit Margin (%)", "RSI", "12M Perf", "Investment Score (1–100)"
]]
z = heatmap_df.values
x = heatmap_df.columns.tolist()
y = heatmap_df.index.tolist()
fig_heatmap = ff.create_annotated_heatmap(
    z=z,
    x=x,
    y=y,
    colorscale='RdBu',
    showscale=True,
    annotation_text=[[f"{val:.2f}" for val in row] for row in z],
    hoverinfo='z'
)
fig_heatmap.update_layout(
    title="Key Financial Metrics per Ticker",
    xaxis_title="Metric",
    yaxis_title="Ticker",
    autosize=True,
    margin=dict(l=40, r=40, t=40, b=40)
)
st.plotly_chart(fig_heatmap, use_container_width=True)

# Bar plot of investment scores
st.subheader("🏆 Investment Score by Ticker")
fig2 = px.bar(
    df.sort_values("Investment Score (1–100)", ascending=False),
    x="Ticker",
    y="Investment Score (1–100)",
    color="Investment Score (1–100)",
    color_continuous_scale="tempo",
    title="Investment Score Ranking",
    labels={"Investment Score (1–100)": "Score"}
)
st.plotly_chart(fig2, use_container_width=True)

# Line chart
st.subheader("📈 5-Year Price Performance")
fig3 = go.Figure()
for ticker, prices in price_data.items():
    fig3.add_trace(go.Scatter(
        x=prices.index,
        y=prices.values,
        mode='lines',
        name=ticker
    ))
fig3.update_layout(
    title="5-Year Stock Price History",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    hovermode="x unified"
)
st.plotly_chart(fig3, use_container_width=True)

# Highlight best growth stock
top_growth = df.sort_values("Rev Growth", ascending=False).iloc[0]["Ticker"]
st.success(f"📈 Best Growth: {top_growth}")
