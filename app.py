
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timezone

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Gold Signal AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# DARK PURPLE THEME
# =========================================================
st.markdown("""
<style>
.stApp {
    background: #0b0913;
    color: #f0edf8;
}
[data-testid="stSidebar"] {
    background: #171127;
    border-right: 1px solid #302442;
}
h1, h2, h3 {
    color: #f5f0ff !important;
}
[data-testid="stMetric"] {
    background: #171127;
    border: 1px solid #302442;
    padding: 16px;
    border-radius: 12px;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #302442;
    border-radius: 8px;
}
div.stButton > button {
    background: #7047eb;
    color: white;
    border: 0;
    border-radius: 8px;
}
div.stButton > button:hover {
    background: #8a68ff;
    color: white;
}
div[data-testid="stAlert"] {
    border-radius: 8px;
}
.small-muted {
    color: #a59ab9;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SETTINGS
# =========================================================
API_BASE = "https://biquote.io/api/XAUUSD/ohlc"

TIMEFRAMES = {
    "M1": "1m",
    "M5": "5m",
    "M15": "15m",
    "M30": "30m",
    "H1": "1h",
    "H4": "4h",
}

STRATEGIES = [
    "Scalping",
    "Engulfing",
    "Support & Resistance",
    "Breakout",
]

# =========================================================
# DATA FETCHING
# =========================================================
@st.cache_data(ttl=10, show_spinner=False)
def fetch_candles(interval, limit=300):
    """
    Fetch XAUUSD OHLC candles from Biquote.
    Returns (DataFrame, error_message).
    """
    try:
        response = requests.get(
            API_BASE,
            params={"interval": interval, "limit": limit},
            headers={"Accept": "application/json"},
            timeout=15,
        )

        if response.status_code != 200:
            return pd.DataFrame(), (
                f"API HTTP {response.status_code}: "
                f"{response.text[:250]}"
            )

        payload = response.json()

        if not isinstance(payload, dict):
            return pd.DataFrame(), "Unexpected API response format."

        bars = payload.get("bars", [])

        if not isinstance(bars, list) or len(bars) == 0:
            return pd.DataFrame(), (
                "The API returned no candle bars. "
                f"Response: {str(payload)[:250]}"
            )

        df = pd.DataFrame(bars)

        required = ["openTime", "open", "high", "low", "close"]
        missing = [column for column in required
                   if column not in df.columns]

        if missing:
            return pd.DataFrame(), (
                f"Missing API fields: {missing}. "
                f"Received fields: {list(df.columns)}"
            )

        df["time"] = pd.to_datetime(
            df["openTime"], utc=True, errors="coerce"
        )

        for column in ["open", "high", "low", "close"]:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )

        df = df.dropna(
            subset=["time", "open", "high", "low", "close"]
        )

        # API may return newest candles first.
        df = (
            df.sort_values("time")
            .drop_duplicates(subset=["time"])
            .reset_index(drop=True)
        )

        # Do not calculate signals from the unfinished candle.
        if "isOpen" in df.columns:
            open_flags = (
                df["isOpen"]
                .astype(str)
                .str.lower()
                .isin(["true", "1"])
            )
            df = df.loc[~open_flags].copy()

        df = df.reset_index(drop=True)

        if df.empty:
            return pd.DataFrame(), (
                "No completed candles are available."
            )

        # Basic OHLC sanity check.
        valid = (
            (df["high"] >= df["low"])
            & (df["high"] >= df["open"])
            & (df["high"] >= df["close"])
            & (df["low"] <= df["open"])
            & (df["low"] <= df["close"])
        )

        df = df.loc[valid].reset_index(drop=True)

        if len(df) < 30:
            return df, (
                f"Only {len(df)} completed candles received; "
                "at least 30 are needed for reliable calculations."
            )

        return df, None

    except requests.exceptions.Timeout:
        return pd.DataFrame(), "Request timed out. Try refreshing."
    except requests.exceptions.RequestException as exc:
        return pd.DataFrame(), f"Network request failed: {exc}"
    except ValueError as exc:
        return pd.DataFrame(), f"Invalid JSON response: {exc}"
    except Exception as exc:
        return pd.DataFrame(), f"Unexpected data error: {exc}"


@st.cache_data(ttl=10, show_spinner=False)
def fetch_live_quote():
    """Fetch the latest available XAUUSD quote."""
    try:
        response = requests.get(
            "https://biquote.io/api/XAUUSD",
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        price = data.get("mid") or data.get("bid") or data.get("ask")
        if price is None:
            return None, "Quote response did not contain a price."

        return float(price), None

    except Exception as exc:
        return None, str(exc)


# =========================================================
# TECHNICAL INDICATORS
# =========================================================
def add_indicators(df):
    df = df.copy()

    df["ema9"] = df["close"].ewm(
        span=9, adjust=False
    ).mean()

    df["ema20"] = df["close"].ewm(
        span=20, adjust=False
    ).mean()

    df["ema50"] = df["close"].ewm(
        span=50, adjust=False
    ).mean()

    df["sma20"] = df["close"].rolling(20).mean()
    df["std20"] = df["close"].rolling(20).std()

    df["bb_upper"] = df["sma20"] + 2 * df["std20"]
    df["bb_lower"] = df["sma20"] - 2 * df["std20"]

    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr14"] = true_range.rolling(14).mean()

    return df


def get_trend(df):
    if len(df) < 50:
        return "Unknown"

    last = df.iloc[-1]

    if pd.isna(last["ema20"]) or pd.isna(last["ema50"]):
        return "Unknown"

    if last["close"] > last["ema20"] > last["ema50"]:
        return "Bullish"

    if last["close"] < last["ema20"] < last["ema50"]:
        return "Bearish"

    return "Sideways"


# =========================================================
# SUPPORT AND RESISTANCE
# =========================================================
def get_support_resistance(df, lookback=40):
    recent = df.tail(lookback)

    if recent.empty:
        return None, None

    support = float(recent["low"].min())
    resistance = float(recent["high"].max())

    return support, resistance


# =========================================================
# SIGNAL ENGINE
# =========================================================
def evaluate_strategy(df, strategy, rr_target=2.0):
    """
    Educational technical-rule engine.
    Returns one dictionary per timeframe/strategy.
    """
    result = {
        "Signal": "WAIT",
        "Trend": get_trend(df),
        "Entry": None,
        "Stop Loss": None,
        "TP1": None,
        "TP2": None,
        "TP3": None,
        "Reason": "Rules not met",
    }

    if len(df) < 55:
        result["Reason"] = "Not enough completed candles"
        return result

    last = df.iloc[-1]
    previous = df.iloc[-2]

    price = float(last["close"])
    atr = last["atr14"]

    if pd.isna(atr) or not np.isfinite(atr) or atr <= 0:
        result["Reason"] = "ATR is unavailable"
        return result

    atr = float(atr)
    trend = result["Trend"]

    bullish_engulfing = (
        previous["close"] < previous["open"]
        and last["close"] > last["open"]
        and last["open"] <= previous["close"]
        and last["close"] >= previous["open"]
    )

    bearish_engulfing = (
        previous["close"] > previous["open"]
        and last["close"] < last["open"]
        and last["open"] >= previous["close"]
        and last["close"] <= previous["open"]
    )

    support, resistance = get_support_resistance(
        df.iloc[:-1], lookback=40
    )

    buy = False
    sell = False
    reason = ""

    # 1. SCALPING
    if strategy == "Scalping":
        buy = (
            last["ema9"] > last["ema20"]
            and last["close"] > last["ema9"]
            and trend == "Bullish"
        )
        sell = (
            last["ema9"] < last["ema20"]
            and last["close"] < last["ema9"]
            and trend == "Bearish"
        )
        reason = "EMA 9/20 momentum with trend filter"

    # 2. ENGULFING
    elif strategy == "Engulfing":
        buy = bullish_engulfing and trend != "Bearish"
        sell = bearish_engulfing and trend != "Bullish"
        reason = "Engulfing candle with trend filter"

    # 3. SUPPORT & RESISTANCE
    elif strategy == "Support & Resistance":
        tolerance = max(atr * 0.35, price * 0.0002)

        buy = (
            support is not None
            and abs(float(last["low"]) - support) <= tolerance
            and last["close"] > last["open"]
        )

        sell = (
            resistance is not None
            and abs(float(last["high"]) - resistance) <= tolerance
            and last["close"] < last["open"]
        )

        reason = "Reaction near recent support/resistance"

    # 4. BREAKOUT
    elif strategy == "Breakout":
        if len(df) >= 22:
            previous_range = df.iloc[-21:-1]
            range_high = float(previous_range["high"].max())
            range_low = float(previous_range["low"].min())

            buy = (
                last["close"] > range_high
                and last["close"] > last["open"]
            )

            sell = (
                last["close"] < range_low
                and last["close"] < last["open"]
            )

            reason = "Close beyond previous 20-candle range"

    if buy and sell:
        result["Reason"] = "Conflicting conditions; no signal"
        return result

    if not buy and not sell:
        result["Reason"] = reason + " — conditions not met"
        return result

    # ATR-based stop and targets.
    # These are theoretical levels, not broker-validated orders.
    stop_distance = max(atr * 1.5, price * 0.0005)

    if buy:
        entry = price
        stop = entry - stop_distance
        direction = 1
        result["Signal"] = "BUY"
    else:
        entry = price
        stop = entry + stop_distance
        direction = -1
        result["Signal"] = "SELL"

    risk = abs(entry - stop)

    result["Entry"] = entry
    result["Stop Loss"] = stop
    result["TP1"] = entry + direction * risk
    result["TP2"] = entry + direction * rr_target * risk
    result["TP3"] = entry + direction * 3 * risk
    result["Reason"] = reason

    return result


# =========================================================
# SIDEBAR CONTROLS
# =========================================================
st.sidebar.title("Gold Signal AI")
st.sidebar.caption("XAUUSD multi-timeframe scanner")

selected_timeframes = st.sidebar.multiselect(
    "Timeframes",
    options=list(TIMEFRAMES.keys()),
    default=list(TIMEFRAMES.keys()),
)

selected_strategies = st.sidebar.multiselect(
    "Strategies",
    options=STRATEGIES,
    default=STRATEGIES,
)

rr_target = st.sidebar.slider(
    "Risk-to-reward target for TP2",
    min_value=1.0,
    max_value=5.0,
    value=2.0,
    step=0.5,
)

st.sidebar.divider()
st.sidebar.subheader("Risk planning")

account_balance = st.sidebar.number_input(
    "Account balance (USD)",
    min_value=0.0,
    value=100.0,
    step=10.0,
)

risk_percent = st.sidebar.slider(
    "Risk per trade (%)",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1,
)

risk_budget = account_balance * risk_percent / 100

st.sidebar.metric(
    "Planned risk budget",
    f"${risk_budget:.2f}",
)

st.sidebar.caption(
    "Risk budget only. Lot sizing is not calculated. "
    "Actual risk depends on broker contract specifications, "
    "spread, slippage and execution."
)

st.sidebar.caption("Auto-refresh: 10 seconds")

# =========================================================
# DASHBOARD HEADER
# =========================================================
st.title("Gold Signal AI")
st.caption(
    "XAUUSD | Multi-timeframe technical scanner | "
    "Signals only — no trades are placed"
)

# =========================================================
# LIVE QUOTE
# =========================================================
price, quote_error = fetch_live_quote()

header_col1, header_col2, header_col3 = st.columns(3)

with header_col1:
    if price is not None:
        st.metric("Latest XAUUSD quote", f"${price:,.2f}")
    else:
        st.metric("Latest XAUUSD quote", "Unavailable")

with header_col2:
    st.metric("Timeframes selected", len(selected_timeframes))

with header_col3:
    st.metric("Strategies selected", len(selected_strategies))

if quote_error:
    st.warning(f"Live quote unavailable: {quote_error}")

st.caption(
    "Latest refresh (UTC): "
    + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
)

st.divider()

# =========================================================
# FETCH ALL SELECTED TIMEFRAMES
# =========================================================
if not selected_timeframes:
    st.info("Select at least one timeframe from the sidebar.")

if not selected_strategies:
    st.info("Select at least one strategy from the sidebar.")

data_by_timeframe = {}
errors_by_timeframe = {}
trend_rows = []
result_rows = []

with st.spinner("Fetching candles and scanning strategies..."):
    for timeframe in selected_timeframes:
        interval = TIMEFRAMES[timeframe]
        candles, error = fetch_candles(interval)

        if error:
            errors_by_timeframe[timeframe] = error

        if candles.empty:
            data_by_timeframe[timeframe] = candles

            for strategy in selected_strategies:
                result_rows.append({
                    "Timeframe": timeframe,
                    "Strategy": strategy,
                    "Signal": "ERROR",
                    "Trend": "Unknown",
                    "Entry": None,
                    "Stop Loss": None,
                    "TP1": None,
                    "TP2": None,
                    "TP3": None,
                    "Reason": error or "No candle data",
                })

            trend_rows.append({
                "Timeframe": timeframe,
                "Trend": "Unknown",
                "Last Close": None,
                "Candles": 0,
                "Status": error or "No candle data",
            })
            continue

        data_by_timeframe[timeframe] = candles
        enriched = add_indicators(candles)
        data_by_timeframe[timeframe] = enriched

        trend_rows.append({
            "Timeframe": timeframe,
            "Trend": get_trend(enriched),
            "Last Close": float(enriched["close"].iloc[-1]),
            "Candles": len(enriched),
            "Status": "OK",
        })

        for strategy in selected_strategies:
            outcome = evaluate_strategy(
                enriched,
                strategy,
                rr_target=rr_target,
            )

            result_rows.append({
                "Timeframe": timeframe,
                "Strategy": strategy,
                "Signal": outcome["Signal"],
                "Trend": outcome["Trend"],
                "Entry": outcome["Entry"],
                "Stop Loss": outcome["Stop Loss"],
                "TP1": outcome["TP1"],
                "TP2": outcome["TP2"],
                "TP3": outcome["TP3"],
                "Reason": outcome["Reason"],
            })

# =========================================================
# DATA WARNINGS
# =========================================================
if errors_by_timeframe:
    with st.expander(
        f"Data warnings ({len(errors_by_timeframe)})",
        expanded=True,
    ):
        for timeframe, error in errors_by_timeframe.items():
            st.warning(f"{timeframe}: {error}")

# =========================================================
# SUMMARY METRICS
# =========================================================
results_df = pd.DataFrame(result_rows)

if not results_df.empty:
    buy_count = int((results_df["Signal"] == "BUY").sum())
    sell_count = int((results_df["Signal"] == "SELL").sum())
    wait_count = int((results_df["Signal"] == "WAIT").sum())
    error_count = int((results_df["Signal"] == "ERROR").sum())
else:
    buy_count = sell_count = wait_count = error_count = 0

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric("BUY setups", buy_count)
metric2.metric("SELL setups", sell_count)
metric3.metric("WAIT", wait_count)
metric4.metric("Data errors", error_count)

# =========================================================
# ALL STRATEGY RESULTS
# =========================================================
st.subheader("All strategy results")

def format_price(value):
    if value is None or pd.isna(value):
        return "—"
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "—"


if not results_df.empty:
    display_df = results_df.copy()

    for column in ["Entry", "Stop Loss", "TP1", "TP2", "TP3"]:
        display_df[column] = display_df[column].apply(format_price)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("Select at least one timeframe and strategy.")

# =========================================================
# DETECTED SETUPS
# =========================================================
st.subheader("Detected setups")

if not results_df.empty:
    signals_df = results_df[
        results_df["Signal"].isin(["BUY", "SELL"])
    ].copy()

    if signals_df.empty:
        if error_count:
            st.warning(
                "No signals can be confirmed while data errors remain. "
                "Check the Data warnings section."
            )
        else:
            st.info(
                "No qualifying setups detected. WAIT means the "
                "selected rules were not met."
            )
    else:
        for _, signal in signals_df.iterrows():
            with st.container(border=True):
                st.markdown(
                    f"**{signal['Signal']} — {signal['Timeframe']} — "
                    f"{signal['Strategy']}**"
                )

                a, b, c, d, e = st.columns(5)
                a.metric("Entry", format_price(signal["Entry"]))
                b.metric("Stop Loss", format_price(signal["Stop Loss"]))
                c.metric("TP1", format_price(signal["TP1"]))
                d.metric("TP2", format_price(signal["TP2"]))
                e.metric("TP3", format_price(signal["TP3"]))

                st.caption(
                    f"Trend: {signal['Trend']} | "
                    f"Reason: {signal['Reason']}"
                )

# =========================================================
# TREND OVERVIEW
# =========================================================
st.subheader("Trend overview")

trend_df = pd.DataFrame(trend_rows)

if not trend_df.empty:
    if "Last Close" in trend_df.columns:
        trend_df["Last Close"] = trend_df["Last Close"].apply(
            format_price
        )

    st.dataframe(
        trend_df,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No timeframe data to display.")

# =========================================================
# CANDLESTICK CHART
# =========================================================
st.subheader("Candlestick chart")

chart_timeframe = st.selectbox(
    "Chart timeframe",
    options=selected_timeframes if selected_timeframes else ["M5"],
    index=0,
)

chart_df = data_by_timeframe.get(
    chart_timeframe, pd.DataFrame()
)

if chart_df.empty:
    st.warning(
        f"No chart available for {chart_timeframe}. "
        "Resolve its data warning above."
    )
else:
    chart_data = chart_df.tail(150).copy()

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=chart_data["time"],
            open=chart_data["open"],
            high=chart_data["high"],
            low=chart_data["low"],
            close=chart_data["close"],
            name="XAUUSD",
            increasing_line_color="#36d6a0",
            decreasing_line_color="#ff5b78",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=chart_data["time"],
            y=chart_data["ema20"],
            mode="lines",
            name="EMA 20",
            line=dict(color="#a78bfa", width=1.4),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=chart_data["time"],
            y=chart_data["ema50"],
            mode="lines",
            name="EMA 50",
            line=dict(color="#f4b860", width=1.4),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0913",
        plot_bgcolor="#110d1c",
        font=dict(color="#e8e0f5"),
        height=600,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.02, x=0),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#292137",
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#292137",
        side="right",
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# FOOTER
# =========================================================
st.divider()

st.caption(
    "Experimental technical-rule scanner. Signals are based on "
    "completed candles and simple indicators. This is not financial "
    "advice. It does not account for broker-specific spreads, slippage, "
    "contract specifications, execution, or news volatility. "
    "Verify data and test the rules before risking real money."
)

# =========================================================
# AUTO REFRESH
# =========================================================
# Streamlit 1.50 supports fragments with automatic reruns.
@st.fragment(run_every="10s")
def refresh_notice():
    st.caption("Automatic refresh is enabled (10 seconds).")

refresh_notice()
