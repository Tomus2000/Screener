# -*- coding: utf-8 -*-
"""INVESTMENT COCKPIT"""

import requests, time
FINNHUB_API_KEY = "d0hiea9r01qup0c6eeugd0hiea9r01qup0c6eev0"

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

st.set_page_config(layout="wide")

st.title("📊 Growth Stock Screener Dashboard")
st.markdown("Insert Portfolio, Analyze and Rebalance. Designed by Tom")

# -------------------------------------------------------
# Stock selection (UNCHANGED)
# -------------------------------------------------------
st.sidebar.header("⚙️ Stock Selection")
tickers_input = st.sidebar.text_input(
    "Enter tickers (comma-separated)",
    value="AXON, CELH, DUOL, INTA, IOT, APP, ENPH, ON, DT, GLOB"
)

st.sidebar.subheader("🔍 Filters")
min_score = st.sidebar.slider("Minimum Investment Score", 1, 100, 1)

# -------------------------------------------------------
# === Portfolio input & overview (NEW UI) ===
# -------------------------------------------------------
st.sidebar.header("📁 Portfolio Input")

# Initialize session state for manual positions
if "manual_positions" not in st.session_state:
    st.session_state.manual_positions = []  # list of dicts: {"Ticker":..., "Buy Price":..., "Quantity":...}

def _add_manual_position(ticker: str, buy_price: float, qty: float):
    if not ticker:
        st.sidebar.warning("Please enter a ticker.")
        return
    if buy_price is None or buy_price <= 0:
        st.sidebar.warning("Buy Price must be positive.")
        return
    if qty is None or qty == 0:
        st.sidebar.warning("Quantity must be non-zero.")
        return
    # normalize ticker
    t = ticker.strip().upper()
    st.session_state.manual_positions.append({"Ticker": t, "Buy Price": float(buy_price), "Quantity": float(qty)})
    st.sidebar.success(f"Added {t} x {qty} @ {buy_price:.2f}")

def _remove_last():
    if st.session_state.manual_positions:
        removed = st.session_state.manual_positions.pop()
        st.sidebar.info(f"Removed last: {removed['Ticker']}")

def _clear_all():
    st.session_state.manual_positions = []
    st.sidebar.info("Cleared manual positions.")

# CSV upload (kept)
uploaded_csv = st.sidebar.file_uploader(
    "Upload portfolio CSV (columns: Ticker, Buy Price, Quantity)",
    type=["csv"]
)

# Beautiful manual entry form (no table)
with st.sidebar.expander("➕ Add Position Manually", expanded=True):
    st.caption("Add a single position at a time. Use the buttons below to manage the list.")
    with st.form("manual_add_form", clear_on_submit=True):
        c1, c2 = st.columns([1,1])
        ticker = c1.text_input("Ticker", placeholder="e.g., AAPL")
        buy_price = c2.number_input("Buy Price", min_value=0.0, step=0.01, format="%.2f", value=0.00)
        qty = st.number_input("Quantity", step=1.0, format="%.2f", value=0.00)
        add_clicked = st.form_submit_button("Add to Portfolio")
        if add_clicked:
            _add_manual_position(ticker, buy_price, qty)

    # Quick actions row
    b1, b2 = st.columns(2)
    with b1:
        if st.button("↩️ Remove Last", use_container_width=True):
            _remove_last()
    with b2:
        if st.button("🗑️ Clear All", use_container_width=True):
            _clear_all()

# Compact preview of manual positions
if st.session_state.manual_positions:
    st.sidebar.markdown("**Current manual positions:**")
    for i, p in enumerate(st.session_state.manual_positions, 1):
        st.sidebar.markdown(
            f"- **{p['Ticker']}** — Qty: {p['Quantity']:.2f} @ {p['Buy Price']:.2f}"
        )

# Normalization helpers
def _clean_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Ticker", "Buy Price", "Quantity"])
    cols = {c.lower().strip(): c for c in df.columns}
    mapping = {}
    for want in ["ticker", "buy price", "quantity"]:
        if want in cols:
            mapping[cols[want]] = want.title()
        else:
            for k, v in cols.items():
                if want.replace(" ","") == k.replace(" ",""):
                    mapping[v] = want.title()
                    break
    df = df.rename(columns=mapping)
    needed = ["Ticker", "Buy Price", "Quantity"]
    for n in needed:
        if n not in df.columns: df[n] = np.nan
    df = df[needed]
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Buy Price"] = pd.to_numeric(df["Buy Price"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df = df.dropna(subset=["Ticker","Buy Price","Quantity"])
    df = df[df["Ticker"]!=""]
    df = df[df["Quantity"]!=0]
    return df

csv_df = None
if uploaded_csv is not None:
    try:
        csv_df = pd.read_csv(uploaded_csv)
    except Exception as e:
        st.sidebar.error(f"Could not read CSV: {e}")

manual_df = pd.DataFrame(st.session_state.manual_positions)
manual_clean = _clean_portfolio(manual_df)
csv_clean = _clean_portfolio(csv_df)
portfolio_input = pd.concat([csv_clean, manual_clean], ignore_index=True)

if not portfolio_input.empty:
    # aggregate duplicates (CSV + manual)
    portfolio_input = (
        portfolio_input
        .groupby("Ticker", as_index=False)
        .agg({"Buy Price":"mean","Quantity":"sum"})
    )

@st.cache_data(ttl=300, show_spinner=False)
def fetch_current_prices(tickers: list) -> pd.Series:
    if not tickers:
        return pd.Series(dtype=float)
    prices = {}
    ts = yf.Tickers(" ".join(tickers))
    for t in tickers:
        p = None
        try:
            p = getattr(ts, t).fast_info.get("last_price", None)
        except Exception:
            p = None
        if p is None:
            try:
                h = getattr(ts, t).history(period="1d")
                if not h.empty:
                    p = float(h["Close"].iloc[-1])
            except Exception:
                p = None
        prices[t] = p if p is not None else np.nan
    return pd.Series(prices, name="Current Price")

# === Portfolio Overview (main area top)
if not portfolio_input.empty:
    st.header("📦 Portfolio Overview")
    current_px = fetch_current_prices(portfolio_input["Ticker"].unique().tolist())
    port = portfolio_input.merge(current_px.rename_axis("Ticker").reset_index(), on="Ticker", how="left")
    port["Cost Basis"] = port["Buy Price"] * port["Quantity"]
    port["Market Value"] = port["Current Price"] * port["Quantity"]
    port["P/L"] = port["Market Value"] - port["Cost Basis"]
    port["P/L %"] = np.where(port["Cost Basis"]>0, port["P/L"]/port["Cost Basis"]*100, np.nan)

    totals = {
        "Total Cost Basis": float(port["Cost Basis"].sum()),
        "Total Market Value": float(port["Market Value"].sum()),
        "Total P/L": float(port["P/L"].sum()),
        "Total P/L %": float(
            (port["Market Value"].sum() - port["Cost Basis"].sum())/port["Cost Basis"].sum()*100
        ) if port["Cost Basis"].sum()>0 else np.nan
    }

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cost Basis", f"${totals['Total Cost Basis']:,.0f}")
    c2.metric("Total Market Value", f"${totals['Total Market Value']:,.0f}")
    c3.metric("Total P/L", f"${totals['Total P/L']:,.0f}",
              delta=f"{totals['Total P/L %']:.2f}%" if pd.notna(totals["Total P/L %"]) else None)
    c4.metric("Positions", f"{len(port)}")

    mv_sum = port["Market Value"].sum()
    port["Weight %"] = np.where(mv_sum>0, port["Market Value"]/mv_sum*100, 0.0)

    st.subheader("🧾 Positions")
    show_cols = ["Ticker","Quantity","Buy Price","Current Price","Cost Basis","Market Value","P/L","P/L %","Weight %"]
    st.dataframe(port[show_cols].set_index("Ticker").round(2), use_container_width=True)

    @st.cache_data
    def _portfolio_csv(df):
        return df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Portfolio Overview (CSV)",
        data=_portfolio_csv(port.round(4)),
        file_name="portfolio_overview.csv",
        mime="text/csv"
    )

# -------------------------------------------------------
# Screener logic (UNCHANGED)
# -------------------------------------------------------
watchlist = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
price_data = {}
results = []

finnhub_url = "https://finnhub.io/api/v1"
def get_finnhub_json(endpoint, params):
    params['token'] = FINNHUB_API_KEY
    r = requests.get(f"{finnhub_url}/{endpoint}", params=params)
    return r.json() if r.status_code == 200 else {}

with st.spinner("Fetching data..."):
    for ticker in watchlist:
        try:
            stock = yf.Ticker(ticker)
            hist_5y = stock.history(period="5y", interval="1d")
            if not hist_5y.empty:
                price_data[ticker] = hist_5y['Close']

            info = stock.info
            pe = info.get("trailingPE")
            eps_growth = info.get("earningsQuarterlyGrowth")
            rev_growth = info.get("revenueGrowth")
            roe = info.get("returnOnEquity")
            dividend_yield = info.get("dividendYield")
            perf_12m = info.get("52WeekChange")

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
            beta = info.get("beta") or fundamentals.get("metric", {}).get("beta")

            peg = (pe / (rev_growth * 100)) if pe and rev_growth else None

            history = stock.history(period="6mo", interval="1d")
            delta = history['Close'].diff()
            gain = delta.clip(lower=0).rolling(window=14).mean()
            loss = -delta.clip(upper=0).rolling(window=14).mean()
            RS = gain / loss
            RSI = 100 - (100 / (1 + RS))
            latest_rsi = RSI.iloc[-1] if not RSI.empty else None

            try:
                latest_earn = earnings[0]
                actual_eps = latest_earn.get("actual")
                estimate_eps = latest_earn.get("estimate")
                earnings_surprise = round((actual_eps - estimate_eps) / estimate_eps * 100, 2) if actual_eps and estimate_eps else 0
            except:
                earnings_surprise = 0

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
                "Profit Margin (%)": round(profit_margin * 100, 2) if profit_margin not in [None, 0] else 0,
                "Beta": round(beta, 2) if beta else None,
                "RSI": round(latest_rsi, 2) if latest_rsi else None,
                "12M Perf": perf_12m,
                "Investment Score (1–100)": round(investment_score, 2),
            })
        except Exception as e:
            st.warning(f"Error with {ticker}: {e}")

df = pd.DataFrame(results).fillna(0)
df = df[df["Investment Score (1–100)"] >= min_score]
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

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Screener Results as CSV",
    data=convert_df(df),
    file_name="growth_screener_results.csv",
    mime="text/csv"
)

st.subheader("🔥 Interactive Heatmap of Key Metrics")
heatmap_df = df.set_index("Ticker")[[
    "Rev Growth","EPS Growth","Earnings Surprise (%)","ROE","Profit Margin (%)","Beta","RSI","12M Perf","Investment Score (1–100)"
]]
z = heatmap_df.values
x = heatmap_df.columns.tolist()
y = heatmap_df.index.tolist()
fig_heatmap = ff.create_annotated_heatmap(z=z, x=x, y=y, colorscale='RdBu',
                                          showscale=True,
                                          annotation_text=[[f"{val:.2f}" for val in row] for row in z],
                                          hoverinfo='z')
fig_heatmap.update_layout(title="Key Financial Metrics per Ticker",
                          xaxis_title="Metric", yaxis_title="Ticker",
                          autosize=True, margin=dict(l=40,r=40,t=40,b=40))
st.plotly_chart(fig_heatmap, use_container_width=True)

st.subheader("🏆 Investment Score by Ticker")
fig2 = px.bar(
    df.sort_values("Investment Score (1–100)", ascending=False),
    x="Ticker", y="Investment Score (1–100)",
    color="Investment Score (1–100)", color_continuous_scale="tempo",
    title="Investment Score Ranking", labels={"Investment Score (1–100)":"Score"}
)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("📈 5-Year Price Performance")
price_data = {}
for t in [t.strip().upper() for t in tickers_input.split(",") if t.strip()]:
    try:
        h = yf.Ticker(t).history(period="5y", interval="1d")
        if not h.empty: price_data[t] = h["Close"]
    except: pass
fig3 = go.Figure()
for t, prices in price_data.items():
    if t not in df["Ticker"].values: continue
    fig3.add_trace(go.Scatter(x=prices.index, y=prices.values, mode='lines', name=t))
fig3.update_layout(title="5-Year Stock Price History", xaxis_title="Date", yaxis_title="Price (USD)", hovermode="x unified")
st.plotly_chart(fig3, use_container_width=True)

if not df.empty:
    top_growth = df.sort_values("Rev Growth", ascending=False).iloc[0]["Ticker"]
    st.success(f"📈 Best Growth: {top_growth}")
