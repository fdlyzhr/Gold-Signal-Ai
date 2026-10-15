
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime

# =========================================================
# GOLD SIGNAL AI — XAUUSD MULTI-TIMEFRAME SCANNER
# =========================================================

st.set_page_config(
    page_title="Gold Signal AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "https://biquote.io/api"
REFRESH_SECONDS = 10
CANDLE_LIMIT = 200

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
# THEME
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #100d1c;
        color: #f4f0ff;
    }
    [data-testid="stSidebar"] {
        background: #19132b;
        border-right: 1px solid #35264f;
    }
    [data-testid="stMetric"] {
        background: #1c1630;
        border: 1px solid #382951;
        padding: 16px;
        border-radius: 12px;
    }
    div[data-testid="stMetricValue"] {
        color: #f5edff;
    }
    h1, h2, h3 {
        color: #eee4ff;
    }
    .stButton > button {
        background: #8154d9;
        color: white;
        border-radius: 8px;
        border: none;
    }
    .stButton > button:hover {
        background: #996ff0;
        color: white;
    }
    div[data-testid="stAlert"] {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# DATA FETCHING
# =========================================================

@st.cache_data(ttl=10, show_spinner=False)
def fetch_quote():
    url = f"{API_BASE}/XAUUSD"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=10, show_spinner=False)
def fetch_candles(interval):
    url = f"{API_BASE}/XAUUSD/ohlc"
    params = {
        "interval": interval,
        "limit": CANDLE_LIMIT,
    }

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    payload = response.json()

    # Accept common response formats.
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
        raise ValueError(
            f"No candle data returned for interval {interval}."
        )

    df = pd.DataFrame(rows)

    # Standardise common field names.
    aliases = {
        "time": ["time", "timestamp", "datetime", "date", "t"],
        "open": ["open", "o"],
        "high": ["high", "h"],
        "low": ["low", "l"],
        "close": ["close", "c"],
        "volume": ["volume", "vol", "v"],
    }

    rename = {}
    lower_columns = {str(col).lower(): col for col in df.columns}

    for standard, candidates in aliases.items():
        for candidate in candidates:
            if candidate in lower_columns:
                rename[lower_columns[candidate]] = standard
                break

    df = df.rename(columns=rename)

    required = ["open", "high", "low", "close"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing candle fields: {', '.join(missing)}. "
            f"Returned fields: {', '.join(map(str, df.columns))}"
        )

    if "time" not in df.columns:
        df["time"] = np.arange(len(df))

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required).reset_index(drop=True)

    if len(df) < 30:
        raise ValueError(
            f"Only {len(df)} candles returned; at least 30 are needed."
        )

    # Convert timestamps when possible.
    try:
        numeric_time = pd.to_numeric(df["time"], errors="coerce")

        if numeric_time.notna().all():
            median_time = numeric_time.median()

            if median_time > 1e12:
                df["time"] = pd.to_datetime(
                    numeric_time, unit="ms", utc=True
                )
            elif median_time > 1e9:
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
        subset=["time"], keep="last"
    ).reset_index(drop=True)

    # Remove a potentially incomplete latest candle.
    # The API's timestamp format may differ, so only remove it
    # when a recognisable timestamp is available.
    if len(df) > 30 and pd.api.types.is_datetime64_any_dtype(df["time"]):
        interval_seconds = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "4h": 14400,
        }.get(interval)

        if interval_seconds:
            now = pd.Timestamp.now(tz="UTC")
            last_time = df["time"].iloc[-1]

            if pd.notna(last_time):
                age = (now - last_time).total_seconds()
                if 0 <= age < interval_seconds:
                    df = df.iloc[:-1].reset_index(drop=True)

    if len(df) < 30:
        raise ValueError(
            "Not enough completed candles for analysis."
        )

    return df


def get_quote_price(payload):
    """Extract a price from common quote response formats."""
    if isinstance(payload, (int, float)):
        return float(payload)

    if isinstance(payload, list) and payload:
        return get_quote_price(payload[0])

    if isinstance(payload, dict):
        for key in ("data", "quote", "result"):
            if key in payload:
                result = get_quote_price(payload[key])
                if result is not None:
                    return result

        for key in ("price", "last", "lastPrice", "close", "bid"):
            if key in payload:
                try:
                    return float(payload[key])
                except (TypeError, ValueError):
                    pass

    return None


# =========================================================
# INDICATORS
# =========================================================

def add_indicators(df):
    df = df.copy()

    df["ema9"] = df["close"].ewm(
        span=9, adjust=False
    ).mean()

    df["ema21"] = df["close"].ewm(
        span=21, adjust=False
    ).mean()

    df["ema50"] = df["close"].ewm(
        span=50, adjust=False
    ).mean()

    df["bb_mid"] = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()

    df["bb_upper"] = df["bb_mid"] + 2 * std
    df["bb_lower"] = df["bb_mid"] - 2 * std

    previous_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - previous_close).abs()
    tr3 = (df["low"] - previous_close).abs()

    df["tr"] = pd.concat(
        [tr1, tr2, tr3], axis=1
    ).max(axis=1)

    df["atr"] = df["tr"].rolling(14).mean()

    return df


def trend_direction(df):
    row = df.iloc[-1]

    if row["ema9"] > row["ema21"] > row["ema50"]:
        return "Bullish"

    if row["ema9"] < row["ema21"] < row["ema50"]:
        return "Bearish"

    return "Sideways"


def is_bullish_engulfing(previous, current):
    previous_bearish = previous["close"] < previous["open"]
    current_bullish = current["close"] > current["open"]

    return (
        previous_bearish
        and current_bullish
        and current["open"] <= previous["close"]
        and current["close"] >= previous["open"]
    )


def is_bearish_engulfing(previous, current):
    previous_bullish = previous["close"] > previous["open"]
    current_bearish = current["close"] < current["open"]

    return (
        previous_bullish
        and current_bearish
        and current["open"] >= previous["close"]
        and current["close"] <= previous["open"]
    )


# =========================================================
# STRATEGY ANALYSIS
# =========================================================

def analyse_strategy(raw_df, strategy, rr_ratio=2.0):
    df = add_indicators(raw_df)
    current = df.iloc[-1]
    previous = df.iloc[-2]

    entry = float(current["close"])
    atr = float(current["atr"]) if pd.notna(current["atr"]) else 0

    if not np.isfinite(atr) or atr <= 0:
        return {
            "signal": "WAIT",
            "reason": "ATR is not available",
            "entry": None,
            "sl": None,
            "tp1": None,
            "tp2": None,
            "tp3": None,
            "trend": trend_direction(df),
        }

    trend = trend_direction(df)
    signal = "WAIT"
    reason = "No qualifying setup"
    support = float(df["low"].tail(20).min())
    resistance = float(df["high"].tail(20).max())

    # Recent swing levels used for the S&R strategy.
    recent_lows = df["low"].iloc[-21:-1]
    recent_highs = df["high"].iloc[-21:-1]

    if len(recent_lows) > 0:
        support = float(recent_lows.min())

    if len(recent_highs) > 0:
        resistance = float(recent_highs.max())

    # -----------------------------------------------------
    # 1. SCALPING
    # EMA direction + candle momentum + Bollinger context.
    # -----------------------------------------------------
    if strategy == "Scalping":
        bullish_candle = current["close"] > current["open"]
        bearish_candle = current["close"] < current["open"]

        if (
            current["ema9"] > current["ema21"]
            and current["close"] > current["ema9"]
            and bullish_candle
        ):
            signal = "BUY"
            reason = "Bullish EMA momentum"

        elif (
            current["ema9"] < current["ema21"]
            and current["close"] < current["ema9"]
            and bearish_candle
        ):
            signal = "SELL"
            reason = "Bearish EMA momentum"

    # -----------------------------------------------------
    # 2. ENGULFING CANDLES
    # -----------------------------------------------------
    elif strategy == "Engulfing":
        if (
            is_bullish_engulfing(previous, current)
            and current["close"] > current["ema21"]
        ):
            signal = "BUY"
            reason = "Bullish engulfing above EMA21"

        elif (
            is_bearish_engulfing(previous, current)
            and current["close"] < current["ema21"]
        ):
            signal = "SELL"
            reason = "Bearish engulfing below EMA21"

    # -----------------------------------------------------
    # 3. SUPPORT & RESISTANCE
    # -----------------------------------------------------
    elif strategy == "Support & Resistance":
        proximity = max(atr * 0.35, entry * 0.0003)

        near_support = abs(float(current["low"]) - support) <= proximity
        near_resistance = (
            abs(float(current["high"]) - resistance) <= proximity
        )

        bullish_rejection = (
            current["close"] > current["open"]
            and current["close"] > support + proximity * 0.2
        )

        bearish_rejection = (
            current["close"] < current["open"]
            and current["close"] < resistance - proximity * 0.2
        )

        if near_support and bullish_rejection:
            signal = "BUY"
            reason = "Bullish rejection near support"

        elif near_resistance and bearish_rejection:
            signal = "SELL"
            reason = "Bearish rejection near resistance"

    # -----------------------------------------------------
    # 4. BREAKOUT
    # Close outside the prior 20-candle range.
    # -----------------------------------------------------
    elif strategy == "Breakout":
        prior = df.iloc[-21:-1]
        prior_high = float(prior["high"].max())
        prior_low = float(prior["low"].min())

        if (
            current["close"] > prior_high
            and current["ema9"] > current["ema21"]
        ):
            signal = "BUY"
            reason = "Close broke above previous range"

        elif (
            current["close"] < prior_low
            and current["ema9"] < current["ema21"]
        ):
            signal = "SELL"
            reason = "Close broke below previous range"

    # -----------------------------------------------------
    # STOP LOSS AND TAKE PROFITS
    # -----------------------------------------------------
    sl = tp1 = tp2 = tp3 = None

    if signal == "BUY":
        swing_low = float(df["low"].tail(10).min())
        sl = min(swing_low, entry - atr * 1.2)

        risk = entry - sl

        if risk <= 0:
            signal = "WAIT"
            reason = "Invalid stop-loss distance"
        else:
            tp1 = entry + risk * rr_ratio * 0.5
            tp2 = entry + risk * rr_ratio
            tp3 = entry + risk * rr_ratio * 1.5

    elif signal == "SELL":
        swing_high = float(df["high"].tail(10).max())
        sl = max(swing_high, entry + atr * 1.2)

        risk = sl - entry

        if risk <= 0:
            signal = "WAIT"
            reason = "Invalid stop-loss distance"
        else:
            tp1 = entry - risk * rr_ratio * 0.5
            tp2 = entry - risk * rr_ratio
            tp3 = entry - risk * rr_ratio * 1.5

    return {
        "signal": signal,
        "reason": reason,
        "entry": entry if signal != "WAIT" else None,
        "sl": sl if signal != "WAIT" else None,
        "tp1": tp1 if signal != "WAIT" else None,
        "tp2": tp2 if signal != "WAIT" else None,
        "tp3": tp3 if signal != "WAIT" else None,
        "trend": trend,
        "support": support,
        "resistance": resistance,
        "atr": atr,
    }


# =========================================================
# SIDEBAR SETTINGS
# =========================================================

st.sidebar.title("Gold Signal AI")
st.sidebar.caption("XAUUSD multi-timeframe scanner")
st.sidebar.divider()

selected_timeframes = st.sidebar.multiselect(
    "Timeframes to scan",
    options=list(TIMEFRAMES.keys()),
    default=list(TIMEFRAMES.keys()),
)

selected_strategies = st.sidebar.multiselect(
    "Strategies to scan",
    options=STRATEGIES,
    default=STRATEGIES,
)

rr_ratio = st.sidebar.slider(
    "Risk-to-reward target",
    min_value=1.0,
    max_value=4.0,
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
    min_value=0.25,
    max_value=2.0,
    value=1.0,
    step=0.25,
)

risk_amount = account_balance * risk_percent / 100

st.sidebar.metric(
    "Planned risk budget",
    f"${risk_amount:.2f}",
)

st.sidebar.caption(
    "Risk budget only. Position size is not calculated because "
    "broker contract specifications and tick values must be verified."
)

st.sidebar.divider()
st.sidebar.caption(
    f"Auto-refresh: every {REFRESH_SECONDS} seconds"
)


# =========================================================
# HEADER
# =========================================================

st.title("Gold Signal AI")
st.caption(
    "XAUUSD | Technical strategy scanner | Signals only — "
    "you place trades manually."
)

# =========================================================
# DASHBOARD
# =========================================================

@st.fragment(run_every=REFRESH_SECONDS)
def render_dashboard():
    # Live quote
    try:
        quote_payload = fetch_quote()
        live_price = get_quote_price(quote_payload)

        if live_price is None:
            st.warning(
                "The quote API returned data in an unrecognised format. "
                "Candlestick analysis may still work."
            )
    except Exception as exc:
        live_price = None
        st.warning(f"Live quote unavailable: {exc}")

    top1, top2, top3 = st.columns(3)

    with top1:
        if live_price is not None:
            st.metric("XAUUSD price", f"${live_price:,.2f}")
        else:
            st.metric("XAUUSD price", "Unavailable")

    with top2:
        st.metric("Timeframes selected", len(selected_timeframes))

    with top3:
        st.metric("Strategies selected", len(selected_strategies))

    st.caption(
        "Last dashboard refresh: "
        + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    if not selected_timeframes or not selected_strategies:
        st.info("Select at least one timeframe and one strategy in the sidebar.")
        return

    st.divider()
    st.subheader("Scanning results")

    results = []
    candle_data = {}
    errors = []

    # Fetch one candle set per timeframe and reuse it for all strategies.
    for timeframe in selected_timeframes:
        interval = TIMEFRAMES[timeframe]

        try:
            raw = fetch_candles(interval)
            candle_data[timeframe] = raw

            for strategy in selected_strategies:
                analysis = analyse_strategy(raw, strategy, rr_ratio)

                results.append({
                    "Timeframe": timeframe,
                    "Strategy": strategy,
                    "Signal": analysis["signal"],
                    "Trend": analysis["trend"],
                    "Entry": analysis["entry"],
                    "Stop Loss": analysis["sl"],
                    "TP1": analysis["tp1"],
                    "TP2": analysis["tp2"],
                    "TP3": analysis["tp3"],
                    "Reason": analysis["reason"],
                })

        except Exception as exc:
            errors.append(f"{timeframe} ({interval}): {exc}")

            for strategy in selected_strategies:
                results.append({
                    "Timeframe": timeframe,
                    "Strategy": strategy,
                    "Signal": "ERROR",
                    "Trend": "Unknown",
                    "Entry": None,
                    "Stop Loss": None,
                    "TP1": None,
                    "TP2": None,
                    "TP3": None,
                    "Reason": str(exc),
                })

    if errors:
        with st.expander("Data warnings", expanded=True):
            for error in errors:
                st.warning(error)

    result_df = pd.DataFrame(results)

    if result_df.empty:
        st.info("No scan results available.")
        return

    buy_count = int((result_df["Signal"] == "BUY").sum())
    sell_count = int((result_df["Signal"] == "SELL").sum())
    wait_count = int((result_df["Signal"] == "WAIT").sum())

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric("BUY setups", buy_count)
    metric2.metric("SELL setups", sell_count)
    metric3.metric("WAIT", wait_count)

    st.markdown("### All strategy results")

    display_df = result_df.copy()

    price_columns = ["Entry", "Stop Loss", "TP1", "TP2", "TP3"]

    for col in price_columns:
        display_df[col] = display_df[col].apply(
            lambda value: f"{value:,.2f}"
            if pd.notna(value) and isinstance(value, (int, float, np.number))
            else "—"
        )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Signal": st.column_config.TextColumn("Signal"),
            "Reason": st.column_config.TextColumn(
                "Setup explanation",
                width="large",
            ),
        },
    )

    # -----------------------------------------------------
    # ACTIVE SETUPS
    # -----------------------------------------------------
    active = result_df[
        result_df["Signal"].isin(["BUY", "SELL"])
    ]

    st.markdown("### Detected setups")

    if active.empty:
        st.info(
            "No setups currently meet the selected rules. "
            "WAIT means the strategy did not detect a qualifying setup."
        )
    else:
        for _, row in active.iterrows():
            direction = row["Signal"]
            heading = (
                f"{direction} · {row['Timeframe']} · {row['Strategy']}"
            )

            with st.expander(heading, expanded=True):
                st.write(f"**Reason:** {row['Reason']}")

                a, b, c = st.columns(3)

                a.metric(
                    "Entry",
                    f"{row['Entry']:,.2f}"
                    if pd.notna(row["Entry"]) else "—",
                )
                b.metric(
                    "Stop Loss",
                    f"{row['Stop Loss']:,.2f}"
                    if pd.notna(row["Stop Loss"]) else "—",
                )
                c.metric(
                    "TP2",
                    f"{row['TP2']:,.2f}"
                    if pd.notna(row["TP2"]) else "—",
                )

                st.write(
                    f"TP1: **{row['TP1']:,.2f}**  |  "
                    f"TP3: **{row['TP3']:,.2f}**"
                    if pd.notna(row["TP1"]) and pd.notna(row["TP3"])
                    else "Take-profit levels unavailable."
                )

    # -----------------------------------------------------
    # TREND OVERVIEW
    # -----------------------------------------------------
    st.divider()
    st.markdown("### Trend overview")

    trend_rows = []

    for timeframe, raw in candle_data.items():
        df = add_indicators(raw)

        trend_rows.append({
            "Timeframe": timeframe,
            "Trend": trend_direction(df),
            "Last completed close": float(df["close"].iloc[-1]),
            "EMA 9": float(df["ema9"].iloc[-1]),
            "EMA 21": float(df["ema21"].iloc[-1]),
            "EMA 50": float(df["ema50"].iloc[-1]),
        })

    if trend_rows:
        trend_df = pd.DataFrame(trend_rows)

        st.dataframe(
            trend_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Last completed close": st.column_config.NumberColumn(
                    format="%.2f"
                ),
                "EMA 9": st.column_config.NumberColumn(format="%.2f"),
                "EMA 21": st.column_config.NumberColumn(format="%.2f"),
                "EMA 50": st.column_config.NumberColumn(format="%.2f"),
            },
        )

    # -----------------------------------------------------
    # CANDLESTICK CHART
    # -----------------------------------------------------
    st.divider()
    st.markdown("### Candlestick chart")

    available_charts = list(candle_data.keys())

    if available_charts:
        chart_timeframe = st.selectbox(
            "Choose chart timeframe",
            available_charts,
            key="chart_timeframe",
        )

        chart_df = add_indicators(
            candle_data[chart_timeframe]
        ).tail(100)

        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=chart_df["time"],
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name="XAUUSD",
        ))

        fig.add_trace(go.Scatter(
            x=chart_df["time"],
            y=chart_df["ema9"],
            name="EMA 9",
            line=dict(width=1.2),
        ))

        fig.add_trace(go.Scatter(
            x=chart_df["time"],
            y=chart_df["ema21"],
            name="EMA 21",
            line=dict(width=1.2),
        ))

        fig.add_trace(go.Scatter(
            x=chart_df["time"],
            y=chart_df["ema50"],
            name="EMA 50",
            line=dict(width=1.2),
        ))

        fig.add_trace(go.Scatter(
            x=chart_df["time"],
            y=chart_df["bb_upper"],
            name="Bollinger Upper",
            line=dict(width=1, dash="dot"),
        ))

        fig.add_trace(go.Scatter(
            x=chart_df["time"],
            y=chart_df["bb_lower"],
            name="Bollinger Lower",
            line=dict(width=1, dash="dot"),
        ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#100d1c",
            plot_bgcolor="#100d1c",
            font=dict(color="#eee4ff"),
            height=600,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "This prototype uses technical rules on completed candles. "
        "It does not account for spread, slippage, news volatility, "
        "broker-specific contract sizing, or execution quality. "
        "Backtest and demo-test before risking money."
    )


render_dashboard()
