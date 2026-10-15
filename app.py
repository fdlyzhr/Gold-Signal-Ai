
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timezone

# =========================================================
# CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Gold Signal AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "https://biquote.io/api/XAUUSD/ohlc"
QUOTE_URL = "https://biquote.io/api/XAUUSD"

TIMEFRAMES = {
    "M1": "1m",
    "M5": "5m",
    "M15": "15m",
    "M30": "30m",
    "H1": "1h",
    "H4": "4h",
}

STRATEGIES = [
    "Trend Pullback",
    "Engulfing",
    "Support & Resistance",
    "Breakout",
]

# =========================================================
# STYLE
# =========================================================
st.markdown("""
<style>
.stApp {background:#0b0913;color:#f0edf8}
[data-testid="stSidebar"] {
    background:#171127;
    border-right:1px solid #302442;
}
h1,h2,h3 {color:#f5f0ff !important}
[data-testid="stMetric"] {
    background:#171127;
    border:1px solid #302442;
    padding:14px;
    border-radius:12px;
}
div[data-testid="stDataFrame"] {
    border:1px solid #302442;
    border-radius:8px;
}
div.stButton>button {
    background:#7047eb;color:white;border:0;border-radius:8px;
}
div.stButton>button:hover {background:#8a68ff;color:white}
</style>
""", unsafe_allow_html=True)


# =========================================================
# MARKET DATA
# =========================================================
@st.cache_data(ttl=10, show_spinner=False)
def fetch_candles(interval, limit=300):
    try:
        r = requests.get(
            API_URL,
            params={"interval": interval, "limit": limit},
            timeout=15,
            headers={"Accept": "application/json"},
        )
        r.raise_for_status()
        payload = r.json()

        bars = payload.get("bars", []) if isinstance(payload, dict) else []
        if not bars:
            return pd.DataFrame(), (
                f"No candles returned for interval {interval}. "
                f"Response: {str(payload)[:180]}"
            )

        df = pd.DataFrame(bars)

        # Accommodate common timestamp field names.
        time_col = next(
            (c for c in ["openTime", "time", "timestamp", "t"]
             if c in df.columns),
            None,
        )

        if time_col is None:
            return pd.DataFrame(), (
                f"No timestamp field. Received: {list(df.columns)}"
            )

        required = ["open", "high", "low", "close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return pd.DataFrame(), f"Missing fields: {missing}"

        raw_time = df[time_col]

        # Unix timestamps may be in seconds or milliseconds.
        if pd.api.types.is_numeric_dtype(raw_time):
            valid_times = raw_time.dropna()
            unit = "ms" if (
                not valid_times.empty
                and valid_times.abs().median() > 1e11
            ) else "s"
            df["time"] = pd.to_datetime(
                raw_time, unit=unit, utc=True, errors="coerce"
            )
        else:
            df["time"] = pd.to_datetime(
                raw_time, utc=True, errors="coerce"
            )

        for c in required:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["time"] + required)
        df = df.sort_values("time")
        df = df.drop_duplicates("time")

        # Remove provider-marked unfinished candles.
        if "isOpen" in df.columns:
            unfinished = (
                df["isOpen"].astype(str).str.lower()
                .isin(["true", "1"])
            )
            df = df.loc[~unfinished]

        # Reject malformed OHLC bars.
        valid = (
            (df["high"] >= df["low"])
            & (df["high"] >= df["open"])
            & (df["high"] >= df["close"])
            & (df["low"] <= df["open"])
            & (df["low"] <= df["close"])
        )
        df = df.loc[valid].reset_index(drop=True)

        if len(df) < 60:
            return df, (
                f"Only {len(df)} candles available; "
                "at least 60 are required."
            )

        return df, None

    except requests.exceptions.Timeout:
        return pd.DataFrame(), "API timed out."
    except requests.exceptions.RequestException as e:
        return pd.DataFrame(), f"Network/API error: {e}"
    except Exception as e:
        return pd.DataFrame(), f"Data error: {e}"


@st.cache_data(ttl=10, show_spinner=False)
def fetch_quote():
    try:
        r = requests.get(QUOTE_URL, timeout=10)
        r.raise_for_status()
        data = r.json()

        for key in ["mid", "price", "last", "bid", "ask"]:
            if data.get(key) is not None:
                return float(data[key]), None

        return None, f"Quote fields not recognized: {str(data)[:180]}"
    except Exception as e:
        return None, str(e)


# =========================================================
# INDICATORS
# =========================================================
def add_indicators(df):
    df = df.copy()

    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    df["sma20"] = df["close"].rolling(20).mean()
    df["std20"] = df["close"].rolling(20).std()
    df["bb_upper"] = df["sma20"] + 2 * df["std20"]
    df["bb_lower"] = df["sma20"] - 2 * df["std20"]

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)

    df["atr"] = tr.rolling(14).mean()

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(
        alpha=1 / 14, adjust=False
    ).mean()
    loss = -delta.clip(upper=0).ewm(
        alpha=1 / 14, adjust=False
    ).mean()

    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Volume is optional; do not fabricate it.
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(
            df["volume"], errors="coerce"
        )

    return df


def trend_of(df):
    if len(df) < 55:
        return "Unknown"

    x = df.iloc[-1]
    if pd.isna(x["ema20"]) or pd.isna(x["ema50"]):
        return "Unknown"

    if x["close"] > x["ema20"] > x["ema50"]:
        return "Bullish"
    if x["close"] < x["ema20"] < x["ema50"]:
        return "Bearish"
    return "Neutral"


def sr_levels(df, lookback=40):
    recent = df.iloc[-lookback-1:-1]
    if len(recent) < 10:
        return None, None
    return float(recent["low"].min()), float(recent["high"].max())


# =========================================================
# HIGH-QUALITY SETUP ENGINE
# =========================================================
def score_setup(df, strategy, higher_trends, min_rr=2.0):
    """
    Returns a setup score and proposed levels.
    The score is a rule-based filter, not a win probability.
    """
    out = {
        "Signal": "WAIT",
        "Score": 0,
        "Quality": "No setup",
        "Trend": trend_of(df),
        "Entry": None,
        "Stop Loss": None,
        "TP1": None,
        "TP2": None,
        "TP3": None,
        "RR": None,
        "Reason": "",
    }

    if len(df) < 60:
        out["Reason"] = "Insufficient candle history"
        return out

    x = df.iloc[-1]
    p = df.iloc[-2]

    price = float(x["close"])
    atr = x["atr"]
    rsi = x["rsi"]

    if pd.isna(atr) or pd.isna(rsi) or atr <= 0:
        out["Reason"] = "Indicators unavailable"
        return out

    atr = float(atr)
    rsi = float(rsi)
    trend = out["Trend"]

    # Higher-timeframe consensus.
    usable_trends = [
        t for t in higher_trends if t in ["Bullish", "Bearish"]
    ]
    bull_count = usable_trends.count("Bullish")
    bear_count = usable_trends.count("Bearish")

    htf_bull = bull_count >= 1 and bull_count > bear_count
    htf_bear = bear_count >= 1 and bear_count > bull_count

    # Avoid very small candles and extreme candle spikes.
    candle_range = float(x["high"] - x["low"])
    body = abs(float(x["close"] - x["open"]))
    body_ratio = body / candle_range if candle_range > 0 else 0

    volatility_ok = 0.0001 * price < atr < 0.02 * price
    candle_size_ok = candle_range < 2.5 * atr

    bull_engulf = (
        p["close"] < p["open"]
        and x["close"] > x["open"]
        and x["open"] <= p["close"]
        and x["close"] >= p["open"]
    )

    bear_engulf = (
        p["close"] > p["open"]
        and x["close"] < x["open"]
        and x["open"] >= p["close"]
        and x["close"] <= p["open"]
    )

    support, resistance = sr_levels(df, 40)
    buy = False
    sell = False
    reasons = []

    # Each strategy defines a potential trigger.
    if strategy == "Trend Pullback":
        buy = (
            trend == "Bullish"
            and htf_bull
            and x["low"] <= x["ema20"] + 0.25 * atr
            and x["close"] > x["ema20"]
            and x["close"] > x["open"]
        )
        sell = (
            trend == "Bearish"
            and htf_bear
            and x["high"] >= x["ema20"] - 0.25 * atr
            and x["close"] < x["ema20"]
            and x["close"] < x["open"]
        )
        reasons.append("EMA20 pullback")

    elif strategy == "Engulfing":
        buy = bull_engulf and trend == "Bullish" and htf_bull
        sell = bear_engulf and trend == "Bearish" and htf_bear
        reasons.append("Engulfing with aligned trend")

    elif strategy == "Support & Resistance":
        tolerance = max(0.35 * atr, price * 0.0002)

        buy = (
            support is not None
            and abs(float(x["low"]) - support) <= tolerance
            and x["close"] > x["open"]
            and trend == "Bullish"
            and htf_bull
        )
        sell = (
            resistance is not None
            and abs(float(x["high"]) - resistance) <= tolerance
            and x["close"] < x["open"]
            and trend == "Bearish"
            and htf_bear
        )
        reasons.append("Support/resistance reaction")

    elif strategy == "Breakout":
        prior = df.iloc[-21:-1]
        range_high = float(prior["high"].max())
        range_low = float(prior["low"].min())

        buy = (
            x["close"] > range_high
            and x["close"] > x["open"]
            and trend == "Bullish"
            and htf_bull
            and body_ratio >= 0.55
        )
        sell = (
            x["close"] < range_low
            and x["close"] < x["open"]
            and trend == "Bearish"
            and htf_bear
            and body_ratio >= 0.55
        )
        reasons.append("Range breakout")

    if buy and sell:
        out["Reason"] = "Conflicting direction"
        return out

    if not buy and not sell:
        out["Reason"] = "No confirmed trigger"
        return out

    direction = "BUY" if buy else "SELL"

    # Reject countertrend setups.
    if direction == "BUY" and not htf_bull:
        out["Reason"] = "Higher timeframes do not confirm BUY"
        return out
    if direction == "SELL" and not htf_bear:
        out["Reason"] = "Higher timeframes do not confirm SELL"
        return out

    # RSI must confirm momentum without being too extended.
    momentum_ok = (
        50 <= rsi <= 70 if direction == "BUY"
        else 30 <= rsi <= 50
    )

    # Avoid weak trigger candles.
    trigger_ok = body_ratio >= 0.40

    if not volatility_ok:
        out["Reason"] = "Volatility outside filter"
        return out
    if not candle_size_ok:
        out["Reason"] = "Signal candle is unusually large"
        return out
    if not momentum_ok:
        out["Reason"] = f"RSI momentum not confirmed ({rsi:.1f})"
        return out
    if not trigger_ok:
        out["Reason"] = "Trigger candle body is too weak"
        return out

    # Score the candidate.
    score = 0
    score += 25  # Trigger exists.
    score += 20 if trend == (
        "Bullish" if direction == "BUY" else "Bearish"
    ) else 0
    score += 20 if (
        htf_bull if direction == "BUY" else htf_bear
    ) else 0
    score += 15 if momentum_ok else 0
    score += 10 if body_ratio >= 0.55 else 5
    score += 10 if volatility_ok and candle_size_ok else 0

    # ATR-based provisional stop. It is not broker-specific.
    entry = price
    stop_distance = max(1.5 * atr, price * 0.0005)
    stop = entry - stop_distance if direction == "BUY" else entry + stop_distance
    risk = abs(entry - stop)

    # TP1, TP2 and TP3 at 1R, 2R and 3R.
    sign = 1 if direction == "BUY" else -1
    tp1 = entry + sign * risk
    tp2 = entry + sign * 2 * risk
    tp3 = entry + sign * 3 * risk
    rr = abs(tp2 - entry) / risk if risk else 0

    if rr < min_rr:
        out["Reason"] = "Reward-to-risk requirement not met"
        return out

    out.update({
        "Signal": direction,
        "Score": score,
        "Quality": (
            "A — Strong filter match" if score >= 85
            else "B — Moderate filter match" if score >= 75
            else "Rejected"
        ),
        "Entry": entry,
        "Stop Loss": stop,
        "TP1": tp1,
        "TP2": tp2,
        "TP3": tp3,
        "RR": rr,
        "Reason": "; ".join(reasons) + f"; RSI {rsi:.1f}",
    })

    return out


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("Gold Signal AI")
st.sidebar.caption("Selective XAUUSD setup scanner")

selected_timeframes = st.sidebar.multiselect(
    "Timeframes",
    list(TIMEFRAMES.keys()),
    default=list(TIMEFRAMES.keys()),
)

selected_strategies = st.sidebar.multiselect(
    "Strategies",
    STRATEGIES,
    default=STRATEGIES,
)

min_score = st.sidebar.slider(
    "Minimum setup score",
    min_value=60,
    max_value=100,
    value=75,
    step=5,
)

min_rr = st.sidebar.slider(
    "Minimum risk-to-reward",
    min_value=1.0,
    max_value=4.0,
    value=2.0,
    step=0.5,
)

st.sidebar.divider()
st.sidebar.subheader("Risk planning")

balance = st.sidebar.number_input(
    "Account balance (USD)",
    min_value=0.0,
    value=100.0,
    step=10.0,
)

risk_pct = st.sidebar.slider(
    "Risk per trade (%)",
    0.1, 5.0, 1.0, 0.1,
)

st.sidebar.metric(
    "Planned risk budget",
    f"${balance * risk_pct / 100:.2f}",
)

st.sidebar.caption(
    "Budget only; not a lot-size calculation. "
    "Verify broker contract specifications."
)

st.sidebar.caption("Auto-refresh: 10 seconds")


# =========================================================
# HEADER / QUOTE
# =========================================================
st.title("Gold Signal AI")
st.caption(
    "High-selectivity technical scanner | Signals only | "
    "No automatic trade execution"
)

price, quote_error = fetch_quote()

c1, c2, c3 = st.columns(3)
c1.metric(
    "Latest XAUUSD quote",
    f"${price:,.2f}" if price is not None else "Unavailable",
)
c2.metric("Selected timeframes", len(selected_timeframes))
c3.metric("Minimum score", f"{min_score}/100")

if quote_error:
    st.warning(f"Live quote unavailable: {quote_error}")

st.caption(
    "Last scan (UTC): "
    + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
)

st.divider()


# =========================================================
# FETCH CANDLES FOR SELECTED TIMEFRAMES
# =========================================================
data = {}
data_errors = {}
trends = {}

for tf in selected_timeframes:
    candles, error = fetch_candles(TIMEFRAMES[tf])

    if error:
        data_errors[tf] = error
        data[tf] = pd.DataFrame()
        trends[tf] = "Unknown"
    else:
        data[tf] = add_indicators(candles)
        trends[tf] = trend_of(data[tf])

# Use H4/H1 as the primary directional context.
# If either context timeframe is unavailable, do not pretend it agrees.
context_trends = [
    trends.get("H4", "Unknown"),
    trends.get("H1", "Unknown"),
]

if "H4" not in selected_timeframes or "H1" not in selected_timeframes:
    context_warning = (
        "Select H1 and H4 for full higher-timeframe confirmation."
    )
else:
    context_warning = None

if context_warning:
    st.info(context_warning)

if data_errors:
    with st.expander(
        f"Data warnings ({len(data_errors)})",
        expanded=True,
    ):
        for tf, message in data_errors.items():
            st.warning(f"{tf}: {message}")


# =========================================================
# SCAN STRATEGIES
# =========================================================
rows = []

for tf in selected_timeframes:
    df = data.get(tf, pd.DataFrame())

    for strategy in selected_strategies:
        if df.empty:
            outcome = {
                "Signal": "ERROR",
                "Score": 0,
                "Quality": "Unavailable",
                "Trend": "Unknown",
                "Entry": None,
                "Stop Loss": None,
                "TP1": None,
                "TP2": None,
                "TP3": None,
                "RR": None,
                "Reason": data_errors.get(tf, "No data"),
            }
        else:
            # Exclude the chart's own timeframe from the context vote.
            other_context = [
                trends.get(context_tf, "Unknown")
                for context_tf in ["H4", "H1"]
                if context_tf != tf
            ]

            # H4/H1 themselves still need the other context timeframe.
            outcome = score_setup(
                df,
                strategy,
                other_context,
                min_rr=min_rr,
            )

            # Strictly require both H4 and H1 to agree for a qualifying setup.
            h4 = trends.get("H4", "Unknown")
            h1 = trends.get("H1", "Unknown")

            if outcome["Signal"] in ["BUY", "SELL"]:
                required_trend = (
                    "Bullish" if outcome["Signal"] == "BUY"
                    else "Bearish"
                )

                if h4 != required_trend or h1 != required_trend:
                    outcome["Signal"] = "WAIT"
                    outcome["Quality"] = "Rejected"
                    outcome["Reason"] = (
                        "H1 and H4 must both agree with direction"
                    )

        # A score is a ranking of rule matches, not a probability.
        if outcome["Signal"] in ["BUY", "SELL"]:
            if outcome["Score"] < min_score:
                outcome["Signal"] = "WAIT"
                outcome["Quality"] = "Rejected"
                outcome["Reason"] = (
                    f"Score {outcome['Score']} below minimum {min_score}"
                )

        rows.append({
            "Timeframe": tf,
            "Strategy": strategy,
            **outcome,
        })

results = pd.DataFrame(rows)


# =========================================================
# SUMMARY
# =========================================================
if results.empty:
    buys = sells = waits = errors = 0
else:
    buys = int((results["Signal"] == "BUY").sum())
    sells = int((results["Signal"] == "SELL").sum())
    waits = int((results["Signal"] == "WAIT").sum())
    errors = int((results["Signal"] == "ERROR").sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Qualified BUY", buys)
m2.metric("Qualified SELL", sells)
m3.metric("WAIT / Rejected", waits)
m4.metric("Data errors", errors)


# =========================================================
# HIGH-QUALITY SETUPS
# =========================================================
st.subheader("High-quality setups")

if results.empty:
    st.info("Choose at least one timeframe and strategy.")
else:
    qualified = results[
        results["Signal"].isin(["BUY", "SELL"])
    ].sort_values("Score", ascending=False)

    if qualified.empty:
        st.info(
            "No setups currently meet your quality filters. "
            "This is expected when conditions are unclear. "
            "Check data warnings before interpreting the result."
        )
    else:
        for _, row in qualified.iterrows():
            with st.container(border=True):
                st.markdown(
                    f"**{row['Signal']} | {row['Timeframe']} | "
                    f"{row['Strategy']}**"
                )

                st.progress(
                    min(max(float(row["Score"]) / 100, 0), 1),
                    text=(
                        f"Rule-match score: {row['Score']}/100 — "
                        f"{row['Quality']}"
                    ),
                )

                a, b, c, d, e = st.columns(5)
                a.metric("Entry", f"{row['Entry']:.2f}")
                b.metric("Stop Loss", f"{row['Stop Loss']:.2f}")
                c.metric("TP1", f"{row['TP1']:.2f}")
                d.metric("TP2", f"{row['TP2']:.2f}")
                e.metric("TP3", f"{row['TP3']:.2f}")

                st.caption(
                    f"Trend: {row['Trend']} | "
                    f"Planned TP2 R:R: {row['RR']:.1f}:1 | "
                    f"{row['Reason']}"
                )


# =========================================================
# ALL SCAN RESULTS
# =========================================================
st.subheader("All strategy results")

if not results.empty:
    display = results.copy()

    for col in ["Entry", "Stop Loss", "TP1", "TP2", "TP3"]:
        display[col] = display[col].apply(
            lambda x: f"{float(x):.2f}"
            if pd.notna(x) else "—"
        )

    display["RR"] = display["RR"].apply(
        lambda x: f"{float(x):.1f}:1"
        if pd.notna(x) else "—"
    )

    st.dataframe(display, use_container_width=True, hide_index=True)


# =========================================================
# TREND OVERVIEW
# =========================================================
st.subheader("Trend overview")

trend_rows = []

for tf in selected_timeframes:
    df = data.get(tf, pd.DataFrame())

    trend_rows.append({
        "Timeframe": tf,
        "Trend": trends.get(tf, "Unknown"),
        "Latest close": (
            round(float(df["close"].iloc[-1]), 2)
            if not df.empty else None
        ),
        "Candles": len(df),
        "Status": "OK" if not df.empty else "ERROR",
    })

if trend_rows:
    st.dataframe(
        pd.DataFrame(trend_rows),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# CANDLESTICK CHART
# =========================================================
st.subheader("Candlestick chart")

if selected_timeframes:
    chart_tf = st.selectbox(
        "Chart timeframe",
        selected_timeframes,
    )

    chart_df = data.get(chart_tf, pd.DataFrame())

    if chart_df.empty:
        st.warning("Chart unavailable because candle data is missing.")
    else:
        chart_df = chart_df.tail(150)

        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=chart_df["time"],
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name="XAUUSD",
            increasing_line_color="#36d6a0",
            decreasing_line_color="#ff5b78",
        ))

        fig.add_trace(go.Scatter(
            x=chart_df["time"],
            y=chart_df["ema20"],
            mode="lines",
            name="EMA 20",
            line=dict(color="#a78bfa", width=1.4),
        ))

        fig.add_trace(go.Scatter(
            x=chart_df["time"],
            y=chart_df["ema50"],
            mode="lines",
            name="EMA 50",
            line=dict(color="#f4b860", width=1.4),
        ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b0913",
            plot_bgcolor="#110d1c",
            font=dict(color="#e8e0f5"),
            height=580,
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=30, b=10),
        )

        fig.update_xaxes(gridcolor="#292137")
        fig.update_yaxes(gridcolor="#292137", side="right")

        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# FOOTER AND AUTO REFRESH
# =========================================================
st.divider()
st.caption(
    "Experimental rules-based scanner. The score is not a win "
    "probability. Signals are not validated trading advice and do "
    "not account for news, spread, slippage, execution or broker "
    "contract specifications. Backtest and demo-test before use."
)

@st.fragment(run_every="10s")
def refresh_status():
    st.caption("Auto-refresh enabled: 10 seconds.")

refresh_status()
