# -*- coding: utf-8 -*-
"""Growth Stock Screener Dashboard with Earnings Surprise from Alpha Vantage"""

import requests
import time

# 🔑 Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = "EAGWCBAFZUOKCABO"
alpha_calls_today = 0
alpha_max_calls = 25  # Free daily limit

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
min_score = st.sidebar.slider("Minimum Investment Score", 1, 10, 1)
min_yield = st.sidebar.slider("Minimum Dividend Yield (%)", 0.0, 10.0, 0.0)

# Convert user input into a list
watchlist = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

price_data = {}
results = []

# Earnings Surprise Function
def get_earnings_surprise(ticker):
    global alpha_calls_today
    if alpha_calls_today >= alpha_max_calls:
        return None

    url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
    try:
        response = requests.get(url)
        alpha_calls_today += 1
        if response.status_code == 200:
            data = response.json()
            q_earnings = data.get("quarterlyEarnings", [])
            if q_earnings:
                latest = q_earnings[0]
                actual = float(latest.get("reportedEPS", 0))
                estimate = float(latest.get("estimatedEPS", 0))
                if estimate != 0:
                    return round((actual - estimate) / estimate * 100, 2)
        return None
    except:
        return None

with st.spinner("Fetching data..."):
    for ticker in watchlist:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            history = stock.history(period="6mo", interval="1d")

            delta = history['Close'].diff()
            gain = delta.clip(lower=0).rolling(window=14).mean()
            loss = -delta.clip(upper=0).rolling(window=14).mean()
            RS = gain / loss
            RSI = 100 - (100 / (1 + RS))
            latest_rsi = RSI.iloc[-1] if not RSI.empty else None

            pe = info.get("trailingPE", None)
            if pe is not None:
                pe = min(pe, 100)
            eps_growth = max(min(info.get("earningsQuarterlyGrowth", 0), 2), -1)
            rev_growth = max(min(info.get("revenueGrowth", 0), 2), -1)
            roe = max(min(info.get("returnOnEquity", 0), 2), -1)
            perf_12m = max(min(info.get("52WeekChange", 0), 2), -1)

            peg = (pe / (rev_growth * 100)) if pe and rev_growth else None

            bs = stock.balance_sheet
            is_ = stock.financials
            try:
                EBIT = is_.loc["EBIT"].iloc[0] if "EBIT" in is_.index else None
                TotalDebt = bs.loc["Long Term Debt"].iloc[0] + bs.loc["Short Long Term Debt"].iloc[0] if "Long Term Debt" in bs.index and "Short Long Term Debt" in bs.index else None
                TotalEquity = bs.loc["Total Stockholder Equity"].iloc[0] if "Total Stockholder Equity" in bs.index else None
                Cash = bs.loc["Cash"] if "Cash" in bs.index else 0
                TaxRate = 0.21
                if EBIT and TotalDebt and TotalEquity:
                    NOPAT = EBIT * (1 - TaxRate)
                    IC = TotalDebt + TotalEquity - Cash.iloc[0] if not isinstance(Cash, (int, float)) else TotalDebt + TotalEquity - Cash
                    roic = NOPAT / IC if IC != 0 else None
                else:
                    roic = None
            except:
                roic = None

            roic = max(min(roic or 0, 2), -1)

            earnings_surprise = get_earnings_surprise(ticker)
            earnings_surprise_score = max(min((earnings_surprise or 0) / 50, 1), -1)

            growth_score = np.mean([rev_growth, eps_growth])
            quality_score = np.mean([roe, roic])
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
                "Company": info.get("shortName", ""),
                "Industry": info.get("industry", ""),
                "PE": pe,
                "PEG": round(peg, 2) if peg else None,
                "Rev Growth": rev_growth,
                "EPS Growth": eps_growth,
                "Earnings Surprise (%)": earnings_surprise,
                "ROE": roe,
                "ROIC": round(roic, 4) if roic else None,
                "RSI": round(latest_rsi, 2) if latest_rsi else None,
                "12M Perf": perf_12m,
                "Investment Score (1–10)": round(investment_score, 2),
                "Dividend Yield (%)": round(info.get("dividendYield", 0) * 100, 2) if info.get("dividendYield") else 0
            })

            hist_5y = stock.history(period="5y", interval="1d")
            if not hist_5y.empty:
                price_data[ticker] = hist_5y['Close']

        except Exception as e:
            st.warning(f"Error with {ticker}: {e}")

    if alpha_calls_today >= alpha_max_calls:
        st.warning("⚠️ Alpha Vantage daily API limit reached (25 calls). Earnings surprise data may be incomplete.")

# Display dataframe
df = pd.DataFrame(results).fillna(0)
st.subheader("📋 Screener Table")
st.dataframe(df.set_index("Ticker"))

st.markdown("""
**📘 Investment Score Explained:**
- **Growth (35%)**: Revenue and EPS growth (YoY).
- **Momentum (20%)**: 12-month price performance.
- **Quality (20%)**: Return on equity and ROIC.
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
heatmap_df = df.set_index("Ticker")[["Rev Growth", "EPS Growth", "Earnings Surprise (%)", "ROE", "ROIC", "RSI", "12M Perf", "Investment Score (1–10)"]]
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
    df.sort_values("Investment Score (1–10)", ascending=False),
    x="Ticker",
    y="Investment Score (1–10)",
    color="Investment Score (1–10)",
    color_continuous_scale="tempo",
    title="Investment Score Ranking",
    labels={"Investment Score (1–10)": "Score"}
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
