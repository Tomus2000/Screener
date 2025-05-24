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

with st.spinner("Fetching data..."):
    for ticker in watchlist:
        try:
            # Use yfinance for price charting only
            stock = yf.Ticker(ticker)
            hist_5y = stock.history(period="5y", interval="1d")
            if not hist_5y.empty:
                price_data[ticker] = hist_5y['Close']

            # Try Yahoo Finance for fundamentals first
            info = stock.info
            pe = info.get("trailingPE")
            eps_growth = info.get("earningsQuarterlyGrowth")
            rev_growth = info.get("revenueGrowth")
            roe = info.get("returnOnEquity")
            dividend_yield = info.get("dividendYield")
            perf_12m = info.get("52WeekChange")

            # If any are None, try Finnhub as fallback
            profile = get_finnhub_json("stock/profile2", {"symbol": ticker})
            fundamentals = get_finnhub_json("stock/metric", {"symbol": ticker, "metric": "all"})
            earnings = get_finnhub_json("stock/earnings", {"symbol": ticker})

            pe = pe if pe is not None else fundamentals.get("metric", {}).get("peNormalizedAnnual")
            eps_growth = eps_growth if eps_growth is not None else fundamentals.get("metric", {}).get("epsGrowth")
            rev_growth = rev_growth if rev_growth is not None else fundamentals.get("metric", {}).get("revenueGrowthYearOverYear")
            roe = roe if roe is not None else fundamentals.get("metric", {}).get("roe")
            dividend_yield = dividend_yield if dividend_yield is not None else fundamentals.get("metric", {}).get("dividendYieldIndicatedAnnual")
            perf_12m = perf_12m if perf_12m is not None else fundamentals.get("metric", {}).get("52WeekPriceReturnDaily")
            profit_margin = fundamentals.get("metric", {}).get("netProfitMarginAnnual")

            # PEG
            peg = (pe / (rev_growth * 100)) if pe and rev_growth else None

            # RSI using yfinance data
            history = stock.history(period="6mo", interval="1d")
            delta = history['Close'].diff()
            gain = delta.clip(lower=0).rolling(window=14).mean()
            loss = -delta.clip(upper=0).rolling(window=14).mean()
            RS = gain / loss
            RSI = 100 - (100 / (1 + RS))
            latest_rsi = RSI.iloc[-1] if not RSI.empty else None

            # Earnings Surprise
            try:
                latest_earn = earnings[0]
                actual_eps = latest_earn.get("actual")
                estimate_eps = latest_earn.get("estimate")
                earnings_surprise = round((actual_eps - estimate_eps) / estimate_eps * 100, 2) if actual_eps and estimate_eps else 0
            except:
                earnings_surprise = 0

            # Scores
            eps_growth = max(min(eps_growth if eps_growth is not None else 0, 2), -1)
            rev_growth = max(min(rev_growth if rev_growth is not None else 0, 2), -1)
            roe = max(min(roe if roe is not None else 0, 2), -1)
            perf_12m = max(min(perf_12m if perf_12m is not None else 0, 2), -1)
            profit_margin = max(min(profit_margin if profit_margin is not None else 0, 2), -1)

            earnings_surprise_score = max(min((earnings_surprise or 0) / 50, 1), -1)
            growth_score = np.mean([rev_growth, eps_growth])
            quality_score = np.mean([roe, profit_margin])
            momentum_score = perf_12m
            valuation_score = max(min((50 - pe) / 50, 1), -1) if pe else 0

            raw_score = (
                0.35 * growth_score +
                0.2 * momentum_score +
                0.2 * quality_score +
                0.15 * earnings_surprise_score +
                0.1 * valuation_score
            )

            investment_score = max(1, min(100, ((raw_score + 1) * 50)))

            results.append({
                "Ticker": ticker,
                "Company": profile.get("name") or info.get("shortName", ""),
                "Industry": profile.get("finnhubIndustry") or info.get("industry", ""),
                "PE": pe,
                "PEG": round(peg, 2) if peg else None,
                "Rev Growth": rev_growth,
                "EPS Growth": eps_growth,
                "Earnings Surprise (%)": earnings_surprise,
                "ROE": roe,
                "Profit Margin (%)": round(profit_margin * 100, 2),
                "RSI": round(latest_rsi, 2) if latest_rsi else None,
                "12M Perf": perf_12m,
                "Investment Score (1–100)": round(investment_score, 2),
                "Dividend Yield (%)": round(dividend_yield * 100, 2) if dividend_yield else 0
            })

        except Exception as e:
            st.warning(f"Error with {ticker}: {e}")

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
