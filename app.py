import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime, timezone

API = "https://biquote.io/api"
REFRESH_SECONDS = 5

st.set_page_config(page_title="Gold Signal AI", page_icon="📈", layout="wide")

st.markdown('''
<style>
.stApp{background:#17151F;color:#F4F0FF}
[data-testid="stHeader"]{background:rgba(23,21,31,.96)}
[data-testid="stSidebar"]{background:#211C2D}
h1,h2,h3,p,label{color:#F4F0FF}
[data-testid="stMetric"]{background:#211C2D;border:1px solid #413451;padding:12px;border-radius:14px}
[data-testid="stMetricValue"]{color:#D8B4FE}
[data-testid="stMetricLabel"]{color:#B9ADC9}
.stButton button{background:#6D4CC3;color:white;border:1px solid #9D7BE8;border-radius:10px;font-weight:700}
hr{border-color:#413451}
.signal-panel{background:#211C2D;border:1px solid #55416E;border-left:4px solid #D8B4FE;border-radius:14px;padding:16px;margin:8px 0 14px}
</style>
''', unsafe_allow_html=True)

@st.cache_data(ttl=5)
def get_quote():
    r = requests.get(f"{API}/XAUUSD", timeout=10)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=5)
def get_candles(interval):
    r = requests.get(f"{API}/XAUUSD/ohlc", params={"interval": interval, "limit": 300}, timeout=15)
    r.raise_for_status()
    payload = r.json()
    if not isinstance(payload, dict) or "bars" not in payload:
        raise ValueError("Unexpected candle data format.")
    df = pd.DataFrame(payload["bars"])
    required = ["openTime", "open", "high", "low", "close"]
    if df.empty or not all(c in df.columns for c in required):
        raise ValueError("Required candle fields are missing.")
    df["openTime"] = pd.to_datetime(df["openTime"], utc=True, errors="coerce")
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required).sort_values("openTime").reset_index(drop=True)
    if "isOpen" in df.columns:
        df["isOpen"] = df["isOpen"].fillna(False).astype(bool)
    else:
        df["isOpen"] = False
    if len(df) < 50:
        raise ValueError("Not enough candles.")
    return df

def completed(df):
    return df[df["isOpen"] != True].copy().reset_index(drop=True) if "isOpen" in df.columns else df.copy()

def indicators(df):
    d = df.copy()
    d["ema9"] = d.close.ewm(span=9, adjust=False).mean()
    d["ema21"] = d.close.ewm(span=21, adjust=False).mean()
    d["ema50"] = d.close.ewm(span=50, adjust=False).mean()
    d["bb_mid"] = d.close.rolling(20).mean()
    sd = d.close.rolling(20).std()
    d["bb_upper"], d["bb_lower"] = d.bb_mid + 2*sd, d.bb_mid - 2*sd
    pc = d.close.shift(1)
    tr = pd.concat([d.high-d.low, (d.high-pc).abs(), (d.low-pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    return d

def analyse(raw, strategy, rr):
    d = indicators(completed(raw))
    if len(d) < 50:
        raise ValueError("Not enough completed candles.")
    last, prev = d.iloc[-1], d.iloc[-2]
    if pd.isna(last.atr) or pd.isna(last.bb_mid):
        raise ValueError("Indicators are not ready.")
    bull_eng = prev.close < prev.open and last.close > last.open and last.open <= prev.close and last.close >= prev.open
    bear_eng = prev.close > prev.open and last.close < last.open and last.open >= prev.close and last.close <= prev.open
    trend = "BULLISH" if last.ema9 > last.ema21 and last.close > last.ema50 else "BEARISH" if last.ema9 < last.ema21 and last.close < last.ema50 else "NEUTRAL"
    price, atr = float(last.close), float(last.atr)
    recent = d.tail(20)
    support, resistance = float(recent.low.min()), float(recent.high.max())
    tolerance = max(atr * .25, price * .00015)
    signal, reason, entry_low, entry_high, stop, targets = "WAIT", "No confirmed setup for this strategy.", None, None, None, []
    if strategy == "Engulfing":
        if bull_eng and trend == "BULLISH":
            signal, reason = "BUY", "Bullish engulfing with bullish EMA trend."
        elif bear_eng and trend == "BEARISH":
            signal, reason = "SELL", "Bearish engulfing with bearish EMA trend."
    elif strategy == "Support & Resistance":
        near_sup = abs(price-support) <= tolerance
        near_res = abs(price-resistance) <= tolerance
        bull_reject = last.close > last.open and last.low <= support+tolerance
        bear_reject = last.close < last.open and last.high >= resistance-tolerance
