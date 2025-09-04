\
import time
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="Premium Trading Tool", layout="wide")
st.title("🚀 Premium Trading Tool")
st.caption("Smart, fast, and simple — made for quick checks by traders.")

BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"  # params: symbol, interval, limit

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(ttl=30)
def fetch_all_tickers():
    r = requests.get(BINANCE_TICKER_URL, timeout=15)
    r.raise_for_status()
    data = pd.DataFrame(r.json())
    # Keep only spot symbols that look like XXXUSDT
    if "symbol" in data.columns:
        data = data[data["symbol"].str.endswith("USDT")]
    # Select and convert
    cols = ["symbol", "lastPrice", "priceChangePercent", "volume"]
    data = data[cols].copy()
    data["lastPrice"] = pd.to_numeric(data["lastPrice"], errors="coerce")
    data["priceChangePercent"] = pd.to_numeric(data["priceChangePercent"], errors="coerce")
    data["volume"] = pd.to_numeric(data["volume"], errors="coerce")
    data = data.dropna().reset_index(drop=True)
    return data.sort_values("volume", ascending=False)

@st.cache_data(ttl=15)
def fetch_klines(symbol: str, limit: int = 60, interval: str = "1m"):
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    r = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
    r.raise_for_status()
    raw = r.json()
    if not isinstance(raw, list) or len(raw) == 0:
        return pd.DataFrame()
    cols = ["open_time","open","high","low","close","volume","close_time","qav","trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    return df[["open_time","close_time","close","volume"]].dropna()

def minute_trend_signal(df: pd.DataFrame) -> dict:
    """
    Very-short-term momentum check using last ~5 minutes slope + last candle change.
    Returns dict with direction, confidence, details.
    """
    result = {"direction": "FLAT", "confidence": 0.0, "last_change_pct": 0.0, "slope": 0.0}
    if df is None or df.empty or len(df) < 3:
        return result
    closes = df["close"].values[-6:] if len(df) >= 6 else df["close"].values
    x = np.arange(len(closes))
    # linear slope
    slope = float(np.polyfit(x, closes, 1)[0])
    # last minute change % vs previous
    last = closes[-1]
    prev = closes[-2] if len(closes) > 1 else closes[-1]
    last_change_pct = 0.0 if prev == 0 else (last - prev) / prev * 100.0
    # direction
    if slope > 0 and last >= prev:
        direction = "UP"
    elif slope < 0 and last <= prev:
        direction = "DOWN"
    else:
        direction = "FLAT"
    # simple confidence: normalized by recent volatility
    recent_range = (np.max(closes) - np.min(closes)) or 1e-9
    confidence = min(1.0, max(0.0, abs(slope) / (recent_range / max(3, len(closes)))))
    return {"direction": direction, "confidence": round(confidence, 2), "last_change_pct": round(last_change_pct, 3), "slope": round(slope, 8)}

# -----------------------------
# Access Gate (Simple)
# -----------------------------
with st.expander("🔑 Premium Access", expanded=True):
    st.info("Monthly premium access required. For demo, tick the checkbox to proceed.")
    paid = st.checkbox("✅ I confirm I have an active premium subscription.")
if not paid:
    st.warning("Please confirm premium access to continue.")
    st.stop()

# -----------------------------
# Auto Refresh
# -----------------------------
colA, colB, colC = st.columns([1,1,2])
with colA:
    do_refresh = st.toggle("Auto-refresh", value=True, help="Reload data automatically.")
with colB:
    every_sec = st.number_input("Refresh (sec)", value=30, min_value=5, max_value=120, step=5)
with colC:
    st.caption("Tip: Lower refresh for faster updates. Binance 24hr ticker updates frequently.")

if do_refresh:
    st.autorefresh(interval=every_sec*1000, key="autorefresh_key")

# -----------------------------
# Live Market Data / Heatmap
# -----------------------------
st.subheader("📊 Live Market (Binance 24hr — USDT pairs)")

try:
    tickers = fetch_all_tickers()
    st.dataframe(tickers.head(50), use_container_width=True)
except Exception as e:
    st.error(f"Failed to load live market data: {e}")
    st.stop()

# Heatmap-style view (Top 20 by volume)
st.markdown("#### 🔥 Top 20 Coins Heatmap (by Volume)")
top20 = tickers.head(20).copy()
# Build a simple heat grid using Plotly bar colored by change %
heat_fig = go.Figure()
heat_fig.add_trace(go.Bar(
    x=top20["symbol"],
    y=top20["priceChangePercent"],
    text=[f'{c:.2f}%' for c in top20["priceChangePercent"]],
    textposition="outside",
    marker=dict(
        color=top20["priceChangePercent"],
        colorscale="RdYlGn",
        cmin=-max(5, abs(top20["priceChangePercent"]).max()),
        cmax=max(5, abs(top20["priceChangePercent"]).max())
    )
))
heat_fig.update_layout(height=350, margin=dict(l=10,r=10,t=10,b=10), xaxis_title="Symbol", yaxis_title="% Change (24h)")
st.plotly_chart(heat_fig, use_container_width=True)

# -----------------------------
# Whale Activity (Top 10 by Volume)
# -----------------------------
st.subheader("🐋 Whale Activity — Top 10 by Volume")
top10 = tickers.sort_values("volume", ascending=False).head(10)
st.dataframe(top10[["symbol","lastPrice","priceChangePercent","volume"]], use_container_width=True)

# -----------------------------
# Gainers & Losers
# -----------------------------
c1, c2 = st.columns(2)
with c1:
    st.markdown("### 📈 Top 5 Gainers")
    gainers = tickers.sort_values("priceChangePercent", ascending=False).head(5)
    st.dataframe(gainers[["symbol","lastPrice","priceChangePercent","volume"]], use_container_width=True)

with c2:
    st.markdown("### 📉 Top 5 Losers")
    losers = tickers.sort_values("priceChangePercent", ascending=True).head(5)
    st.dataframe(losers[["symbol","lastPrice","priceChangePercent","volume"]], use_container_width=True)

# -----------------------------
# Search & 1‑Minute Momentum Signal
# -----------------------------
st.subheader("🔍 Search Coin + 1‑Minute Momentum Signal")
search_col1, search_col2 = st.columns([2,1])
with search_col1:
    symbol = st.text_input("Enter pair (e.g., BTCUSDT, ETHUSDT):", value="BTCUSDT").upper().strip()
with search_col2:
    limit = st.slider("Minutes to show", 10, 120, 60)

if symbol:
    try:
        kdf = fetch_klines(symbol, limit=limit, interval="1m")
        if kdf.empty:
            st.warning("No klines returned. Check the symbol (e.g. BTCUSDT).")
        else:
            # Price line
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=kdf["close_time"], y=kdf["close"], mode="lines", name="Close"))
            fig.update_layout(title=f"{symbol} — Last {len(kdf)} minutes (1m closes)",
                              xaxis_title="Time", yaxis_title="Price",
                              height=400, margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)

            sig = minute_trend_signal(kdf)
            dir_emoji = "🟢" if sig["direction"] == "UP" else ("🔴" if sig["direction"] == "DOWN" else "🟡")
            st.markdown(f"**1‑min Momentum:** {dir_emoji} **{sig['direction']}**  • Confidence: **{sig['confidence']}**  • Last candle change: **{sig['last_change_pct']}%**")
            st.caption("Note: This is a simple momentum heuristic using recent 1‑minute data (not financial advice).")
    except Exception as e:
        st.error(f"Error fetching klines for {symbol}: {e}")
else:
    st.info("Type a symbol like BTCUSDT to see price line and 1‑minute signal.")
