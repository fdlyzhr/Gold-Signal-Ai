
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Gold Signal AI", page_icon="📈", layout="wide")

API = "https://biquote.io/api"
REFRESH = 10
LIMIT = 200

TIMEFRAMES = {
    "M1": "1m", "M5": "5m", "M15": "15m",
    "M30": "30m", "H1": "1h", "H4": "4h"
}
STRATEGIES = [
    "Scalping", "Engulfing",
    "Support & Resistance", "Breakout"
]

# ---------------- THEME ----------------

st.markdown("""
<style>
.stApp {background:#100d1c;color:#f4f0ff}
[data-testid="stSidebar"] {background:#19132b;border-right:1px solid #35264f}
[data-testid="stMetric"] {background:#1c1630;border:1px solid #382951;padding:14px;border-radius:12px}
h1,h2,h3 {color:#eee4ff}
</style>
""", unsafe_allow_html=True)


# ---------------- MARKET DATA ----------------

@st.cache_data(ttl=10, show_spinner=False)
def fetch_quote():
    r = requests.get(f"{API}/XAUUSD", timeout=10)
    r.raise_for_status()
    return r.json()


def extract_price(data):
    if isinstance(data, (float, int)):
        return float(data)
    if isinstance(data, list) and data:
        return extract_price(data[0])
    if isinstance(data, dict):
        for key in ("data", "quote", "result"):
            if key in data:
                price = extract_price(data[key])
                if price is not None:
                    return price
        for key in ("price", "last", "lastPrice", "close", "bid"):
            try:
                if key in data:
                    return float(data[key])
            except (TypeError, ValueError):
                pass
    return None


@st.cache_data(ttl=10, show_spinner=False)
def fetch_candles(interval):
    r = requests.get(
        f"{API}/XAUUSD/ohlc",
        params={"interval": interval, "limit": LIMIT},
        timeout=15
    )
    r.raise_for_status()
    payload = r.json()

    if isinstance(payload, dict):
        rows = (
            payload.get("data")
            or payload.get("candles")
            or payload.get("ohlc")
            or payload.get("results")
        )
    else:
        rows = payload

    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No candle data for interval {interval}")

    df = pd.DataFrame(rows)
    aliases = {
        "time": ["time", "timestamp", "datetime", "date", "t"],
        "open": ["open", "o"],
        "high": ["high", "h"],
        "low": ["low", "l"],
        "close": ["close", "c"]
    }

    columns = {str(c).lower(): c for c in df.columns}
    rename = {}

    for standard, options in aliases.items():
        for option in options:
            if option in columns:
                rename[columns[option]] = standard
                break

    df = df.rename(columns=rename)

    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"API response is missing '{col}'")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "time" not in df.columns:
        df["time"] = np.arange(len(df))

    df = df.dropna(subset=["open", "high", "low", "close"])

    try:
        numeric_time = pd.to_numeric(df["time"], errors="coerce")
        if numeric_time.notna().all():
            median = numeric_time.median()
            if median > 1e12:
                df["time"] = pd.to_datetime(
                    numeric_time, unit="ms", utc=True
                )
            elif median > 1e9:
                df["time"] = pd.to_datetime(
                    numeric_time, unit="s", utc=True
                )
        else:
            df["time"] = pd.to_datetime(
                df["time"], errors="coerce", utc=True
            )
    except Exception:
        pass

    df = df.sort_values("time").drop_duplicates(
        "time", keep="last"
    ).reset_index(drop=True)

    # Remove the latest candle if it appears to be incomplete.
    seconds = {
        "1m": 60, "5m": 300, "15m": 900,
        "30m": 1800, "1h": 3600, "4h": 14400
    }.get(interval)

    if (
        len(df) > 30
        and seconds
        and pd.api.types.is_datetime64_any_dtype(df["time"])
    ):
        age = (
            pd.Timestamp.now(tz="UTC") - df["time"].iloc[-1]
        ).total_seconds()
        if 0 <= age < seconds:
            df = df.iloc[:-1].reset_index(drop=True)

    if len(df) < 30:
        raise ValueError("Fewer than 30 usable candles were returned")

    return df


# ---------------- INDICATORS ----------------

def indicators(data):
    df = data.copy()

    for span in (9, 21, 50):
        df[f"ema{span}"] = df["close"].ewm(
            span=span, adjust=False
        ).mean()

    df["bb_mid"] = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * std
    df["bb_lower"] = df["bb_mid"] - 2 * std

    prev_close = df["close"].shift(1)
    df["tr"] = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)

    df["atr"] = df["tr"].rolling(14).mean()
    return df


def get_trend(df):
    c = df.iloc[-1]

    if c["ema9"] > c["ema21"] > c["ema50"]:
        return "Bullish"
    if c["ema9"] < c["ema21"] < c["ema50"]:
        return "Bearish"
    return "Sideways"


def bullish_engulfing(p, c):
    return (
        p["close"] < p["open"]
        and c["close"] > c["open"]
        and c["open"] <= p["close"]
        and c["close"] >= p["open"]
    )


def bearish_engulfing(p, c):
    return (
        p["close"] > p["open"]
        and c["close"] < c["open"]
        and c["open"] >= p["close"]
        and c["close"] <= p["open"]
    )


# ---------------- STRATEGIES ----------------

def analyse(raw, strategy, rr):
    df = indicators(raw)
    c, p = df.iloc[-1], df.iloc[-2]
    entry = float(c["close"])
    atr = float(c["atr"]) if pd.notna(c["atr"]) else 0
    trend = get_trend(df)

    result = {
        "Signal": "WAIT", "Trend": trend,
        "Entry": None, "Stop Loss": None,
        "TP1": None, "TP2": None, "TP3": None,
        "Reason": "No qualifying setup"
    }

    if not np.isfinite(atr) or atr <= 0:
        result["Reason"] = "ATR unavailable"
        return result

    signal, reason = "WAIT", "No qualifying setup"
    support = float(df["low"].iloc[-21:-1].min())
    resistance = float(df["high"].iloc[-21:-1].max())

    # 1. Scalping: EMA direction and candle momentum
    if strategy == "Scalping":
        if (
            c["ema9"] > c["ema21"]
            and c["close"] > c["ema9"]
            and c["close"] > c["open"]
        ):
            signal, reason = "BUY", "Bullish EMA momentum"

        elif (
            c["ema9"] < c["ema21"]
            and c["close"] < c["ema9"]
            and c["close"] < c["open"]
        ):
            signal, reason = "SELL", "Bearish EMA momentum"

    # 2. Engulfing candle with EMA confirmation
    elif strategy == "Engulfing":
        if bullish_engulfing(p, c) and c["close"] > c["ema21"]:
            signal, reason = "BUY", "Bullish engulfing above EMA21"

        elif bearish_engulfing(p, c) and c["close"] < c["ema21"]:
            signal, reason = "SELL", "Bearish engulfing below EMA21"

    # 3. Support and resistance rejection
    elif strategy == "Support & Resistance":
        distance = max(atr * 0.35, entry * 0.0003)

        near_support = abs(float(c["low"]) - support) <= distance
        near_resistance = abs(float(c["high"]) - resistance) <= distance

        if (
            near_support
            and c["close"] > c["open"]
            and c["close"] > support + distance * 0.2
        ):
            signal, reason = "BUY", "Bullish rejection near support"

        elif (
            near_resistance
            and c["close"] < c["open"]
            and c["close"] < resistance - distance * 0.2
        ):
            signal, reason = "SELL", "Bearish rejection near resistance"

    # 4. Breakout from previous 20-candle range
    elif strategy == "Breakout":
        prior = df.iloc[-21:-1]
        high = float(prior["high"].max())
        low = float(prior["low"].min())

        if c["close"] > high and c["ema9"] > c["ema21"]:
            signal, reason = "BUY", "Breakout above previous range"

        elif c["close"] < low and c["ema9"] < c["ema21"]:
            signal, reason = "SELL", "Breakout below previous range"

    # SL based on recent swing and ATR; targets use selected RR.
    if signal == "BUY":
        sl = min(float(df["low"].tail(10).min()), entry - atr * 1.2)
        risk = entry - sl

        if risk > 0:
            result.update({
                "Signal": signal, "Reason": reason,
                "Entry": entry, "Stop Loss": sl,
                "TP1": entry + risk * rr * 0.5,
                "TP2": entry + risk * rr,
                "TP3": entry + risk * rr * 1.5
            })

    elif signal == "SELL":
        sl = max(float(df["high"].tail(10).max()), entry + atr * 1.2)
        risk = sl - entry

        if risk > 0:
            result.update({
                "Signal": signal, "Reason": reason,
                "Entry": entry, "Stop Loss": sl,
                "TP1": entry - risk * rr * 0.5,
                "TP2": entry - risk * rr,
                "TP3": entry - risk * rr * 1.5
            })

    return result


# ---------------- SIDEBAR SETTINGS ----------------

st.sidebar.title("Gold Signal AI")
st.sidebar.caption("XAUUSD multi-timeframe scanner")

timeframes = st.sidebar.multiselect(
    "Timeframes", list(TIMEFRAMES),
    default=list(TIMEFRAMES)
)

strategies = st.sidebar.multiselect(
    "Strategies", STRATEGIES,
    default=STRATEGIES
)

rr = st.sidebar.slider(
    "Risk-to-reward target", 1.0, 4.0, 2.0, 0.5
)

st.sidebar.divider()
st.sidebar.subheader("Risk planning")

balance = st.sidebar.number_input(
    "Account balance (USD)", min_value=0.0,
    value=100.0, step=10.0
)

risk_pct = st.sidebar.slider(
    "Risk per trade (%)", 0.25, 2.0, 1.0, 0.25
)

st.sidebar.metric(
    "Planned risk budget",
    f"${balance * risk_pct / 100:.2f}"
)

st.sidebar.caption(
    "This is a risk budget only. Lot sizing is not calculated; "
    "broker contract specifications must be verified."
)
st.sidebar.caption(f"Auto-refresh: {REFRESH} seconds")

st.title("Gold Signal AI")
st.caption(
    "XAUUSD | Technical scanner | Signals only — trades are placed manually."
)


# ---------------- LIVE DASHBOARD ----------------

@st.fragment(run_every=REFRESH)
def dashboard():
    try:
        price = extract_price(fetch_quote())
    except Exception as exc:
        price = None
        st.warning(f"Live quote unavailable: {exc}")

    a, b, c = st.columns(3)
    a.metric("XAUUSD price", f"${price:,.2f}" if price is not None else "Unavailable")
    b.metric("Timeframes", len(timeframes))
    c.metric("Strategies", len(strategies))

    st.caption("Updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if not timeframes or not strategies:
        st.info("Select at least one timeframe and one strategy.")
        return

    rows, candles, errors = [], {}, []

    for tf in timeframes:
        try:
            raw = fetch_candles(TIMEFRAMES[tf])
            candles[tf] = raw

            for strategy in strategies:
                result = analyse(raw, strategy, rr)
                rows.append({
                    "Timeframe": tf,
                    "Strategy": strategy,
                    **result
                })

        except Exception as exc:
            errors.append(f"{tf} ({TIMEFRAMES[tf]}): {exc}")
            for strategy in strategies:
                rows.append({
                    "Timeframe": tf, "Strategy": strategy,
                    "Signal": "ERROR", "Trend": "Unknown",
                    "Entry": None, "Stop Loss": None,
                    "TP1": None, "TP2": None, "TP3": None,
                    "Reason": str(exc)
                })

    if errors:
        with st.expander("Data warnings", expanded=True):
            for error in errors:
                st.warning(error)

    table = pd.DataFrame(rows)
    if table.empty:
        st.info("No scan results.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("BUY setups", int((table["Signal"] == "BUY").sum()))
    m2.metric("SELL setups", int((table["Signal"] == "SELL").sum()))
    m3.metric("WAIT", int((table["Signal"] == "WAIT").sum()))

    st.subheader("All strategy results")

    shown = table.copy()
    for col in ["Entry", "Stop Loss", "TP1", "TP2", "TP3"]:
        shown[col] = shown[col].apply(
            lambda v: f"{v:,.2f}"
            if pd.notna(v) and isinstance(v, (int, float, np.number))
            else "—"
        )

    st.dataframe(shown, use_container_width=True, hide_index=True)

    st.subheader("Detected setups")
    active = table[table["Signal"].isin(["BUY", "SELL"])]

    if active.empty:
        st.info("No qualifying setups detected. WAIT means the rules were not met.")
    else:
        for _, row in active.iterrows():
            with st.expander(
                f"{row['Signal']} · {row['Timeframe']} · {row['Strategy']}"
            ):
                st.write("**Reason:**", row["Reason"])
                x, y, z = st.columns(3)
                x.metric("Entry", f"{row['Entry']:,.2f}")
                y.metric("Stop Loss", f"{row['Stop Loss']:,.2f}")
                z.metric("TP2", f"{row['TP2']:,.2f}")
                st.write(
                    f"TP1: **{row['TP1']:,.2f}** | "
                    f"TP3: **{row['TP3']:,.2f}**"
                )

    st.subheader("Trend overview")
    trend_rows = []

    for tf, raw in candles.items():
        df = indicators(raw)
        trend_rows.append({
            "Timeframe": tf,
            "Trend": get_trend(df),
            "Last close": float(df["close"].iloc[-1]),
            "EMA 9": float(df["ema9"].iloc[-1]),
            "EMA 21": float(df["ema21"].iloc[-1]),
            "EMA 50": float(df["ema50"].iloc[-1])
        })

    if trend_rows:
        st.dataframe(
            pd.DataFrame(trend_rows),
            use_container_width=True,
            hide_index=True
        )

    st.subheader("Candlestick chart")

    if candles:
        chart_tf = st.selectbox(
            "Chart timeframe", list(candles),
            key="chart_timeframe"
        )
        chart = indicators(candles[chart_tf]).tail(100)

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=chart["time"],
            open=chart["open"], high=chart["high"],
            low=chart["low"], close=chart["close"],
            name="XAUUSD"
        ))

        for column, label in [
            ("ema9", "EMA 9"),
            ("ema21", "EMA 21"),
            ("ema50", "EMA 50"),
            ("bb_upper", "Bollinger Upper"),
            ("bb_lower", "Bollinger Lower")
        ]:
            fig.add_trace(go.Scatter(
                x=chart["time"], y=chart[column],
                name=label, line=dict(width=1)
            ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#100d1c",
            plot_bgcolor="#100d1c",
            font=dict(color="#eee4ff"),
            height=600,
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Experimental technical rules. Signals do not account for spread, "
        "slippage, news volatility, or broker-specific contract sizing. "
        "Demo-test before risking money."
    )


dashboard()
