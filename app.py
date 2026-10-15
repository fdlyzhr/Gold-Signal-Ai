
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime, timezone

API = "https://biquote.io/api"
REFRESH_SECONDS = 5

st.set_page_config(
    page_title="Gold Signal AI",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
.stApp {
    background: #17151F;
    color: #F4F0FF;
}
[data-testid="stHeader"] {
    background: rgba(23, 21, 31, 0.96);
}
[data-testid="stSidebar"] {
    background: #211C2D;
}
h1, h2, h3, p, label {
    color: #F4F0FF;
}
[data-testid="stMetric"] {
    background: #211C2D;
    border: 1px solid #413451;
    padding: 12px;
    border-radius: 14px;
}
[data-testid="stMetricValue"] {
    color: #D8B4FE;
}
[data-testid="stMetricLabel"] {
    color: #B9ADC9;
}
.stButton button {
    background: #6D4CC3;
    color: white;
    border: 1px solid #9D7BE8;
    border-radius: 10px;
    font-weight: 700;
}
hr {
    border-color: #413451;
}
.signal-panel {
    background: #211C2D;
    border: 1px solid #55416E;
    border-left: 4px solid #D8B4FE;
    border-radius: 14px;
    padding: 16px;
    margin: 8px 0 14px;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# MARKET DATA
# -----------------------------

@st.cache_data(ttl=5)
def get_quote():
    response = requests.get(
        f"{API}/XAUUSD",
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=5)
def get_candles(interval):
    response = requests.get(
        f"{API}/XAUUSD/ohlc",
        params={
            "interval": interval,
            "limit": 300,
        },
        timeout=15,
    )
    response.raise_for_status()

    payload = response.json()

    if not isinstance(payload, dict) or "bars" not in payload:
        raise ValueError("Unexpected candle data format.")

    df = pd.DataFrame(payload["bars"])

    required = [
        "openTime", "open", "high", "low", "close"
    ]

    if df.empty or not all(c in df.columns for c in required):
        raise ValueError("Required candle fields are missing.")

    df["openTime"] = pd.to_datetime(
        df["openTime"],
        utc=True,
        errors="coerce",
    )

    for column in ["open", "high", "low", "close"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(subset=required)
    df = df.sort_values("openTime").reset_index(drop=True)

    if "isOpen" in df.columns:
        df["isOpen"] = (
            df["isOpen"].fillna(False).astype(bool)
        )
    else:
        df["isOpen"] = False

    if len(df) < 50:
        raise ValueError("Not enough candles received.")

    return df


def completed_candles(df):
    if "isOpen" in df.columns:
        return (
            df[df["isOpen"] != True]
            .copy()
            .reset_index(drop=True)
        )
    return df.copy().reset_index(drop=True)


# -----------------------------
# TECHNICAL INDICATORS
# -----------------------------

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

    standard_deviation = df["close"].rolling(20).std()

    df["bb_upper"] = (
        df["bb_mid"] + 2 * standard_deviation
    )

    df["bb_lower"] = (
        df["bb_mid"] - 2 * standard_deviation
    )

    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr"] = true_range.rolling(14).mean()

    return df


# -----------------------------
# STRATEGY ANALYSIS
# -----------------------------

def analyse(raw_df, strategy, rr):
    df = add_indicators(completed_candles(raw_df))

    if len(df) < 50:
        raise ValueError(
            "Not enough completed candles for analysis."
        )

    last = df.iloc[-1]
    previous = df.iloc[-2]

    if pd.isna(last["atr"]) or pd.isna(last["bb_mid"]):
        raise ValueError("Indicators are not ready.")

    price = float(last["close"])
    atr = float(last["atr"])

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

    if (
        last["ema9"] > last["ema21"]
        and last["close"] > last["ema50"]
    ):
        trend = "BULLISH"
    elif (
        last["ema9"] < last["ema21"]
        and last["close"] < last["ema50"]
    ):
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"

    # Approximate recent support and resistance.
    recent = df.tail(20)

    support = float(recent["low"].min())
    resistance = float(recent["high"].max())

    tolerance = max(
        atr * 0.25,
        price * 0.00015,
    )

    signal = "WAIT"
    reason = "No confirmed setup for this strategy."

    entry_low = None
    entry_high = None
    stop_loss = None
    targets = []

    # Strategy 1: Engulfing candle + EMA trend.
    if strategy == "Engulfing":
        if bullish_engulfing and trend == "BULLISH":
            signal = "BUY"
            reason = (
                "Bullish engulfing candle with bullish EMA trend."
            )

        elif bearish_engulfing and trend == "BEARISH":
            signal = "SELL"
            reason = (
                "Bearish engulfing candle with bearish EMA trend."
            )

    # Strategy 2: Support and resistance rejection.
    elif strategy == "Support & Resistance":
        near_support = abs(price - support) <= tolerance
        near_resistance = abs(price - resistance) <= tolerance

        bullish_rejection = (
            last["close"] > last["open"]
            and last["low"] <= support + tolerance
        )

        bearish_rejection = (
            last["close"] < last["open"]
            and last["high"] >= resistance - tolerance
        )

        if (
            near_support
            and bullish_rejection
            and trend != "BEARISH"
        ):
            signal = "BUY"
            reason = (
                "Bullish candle rejection near recent support."
            )

        elif (
            near_resistance
            and bearish_rejection
            and trend != "BULLISH"
        ):
            signal = "SELL"
            reason = (
                "Bearish candle rejection near recent resistance."
            )

    # Strategy 3: Breakout beyond the previous 20-candle range.
    elif strategy == "Breakout":
        prior = df.iloc[-21:-1]

        range_high = float(prior["high"].max())
        range_low = float(prior["low"].min())

        if (
            price > range_high
            and last["close"] > last["open"]
            and trend == "BULLISH"
        ):
            signal = "BUY"
            reason = (
                f"Completed candle closed above resistance "
                f"at {range_high:.2f}."
            )

        elif (
            price < range_low
            and last["close"] < last["open"]
            and trend == "BEARISH"
        ):
            signal = "SELL"
            reason = (
                f"Completed candle closed below support "
                f"at {range_low:.2f}."
            )

    # Entry, stop loss and targets.
    if signal in ("BUY", "SELL"):
        entry_low = price - tolerance / 2
        entry_high = price + tolerance / 2

        if signal == "BUY":
            recent_low = float(df.tail(10)["low"].min())

            stop_loss = min(
                recent_low,
                price - 1.2 * atr,
            )

            risk = price - stop_loss

            targets = [
                price + risk * rr * fraction
                for fraction in (0.5, 1.0, 1.5)
            ]

        else:
            recent_high = float(df.tail(10)["high"].max())

            stop_loss = max(
                recent_high,
                price + 1.2 * atr,
            )

            risk = stop_loss - price

            targets = [
                price - risk * rr * fraction
                for fraction in (0.5, 1.0, 1.5)
            ]

        if risk <= 0:
            signal = "WAIT"
            reason = "Invalid stop distance. Wait for another setup."
            stop_loss = None
            targets = []
            entry_low = None
            entry_high = None

    return {
        "signal": signal,
        "trend": trend,
        "price": price,
        "support": support,
        "resistance": resistance,
        "atr": atr,
        "bb_upper": float(last["bb_upper"]),
        "bb_mid": float(last["bb_mid"]),
        "bb_lower": float(last["bb_lower"]),
        "entry_low": entry_low,
        "entry_high": entry_high,
        "stop": stop_loss,
        "targets": targets,
        "reason": reason,
        "time": last["openTime"],
    }, df


# -----------------------------
# CHARTS
# -----------------------------

def draw_chart(raw_df, title, live_price, analysis):
    df = add_indicators(raw_df)

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df["openTime"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="XAUUSD",
            increasing_line_color="#D8B4FE",
            increasing_fillcolor="#6D4CC3",
            decreasing_line_color="#FCA5A5",
            decreasing_fillcolor="#7F1D1D",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["openTime"],
            y=df["ema9"],
            name="EMA 9",
            line=dict(color="#F0ABFC", width=1.3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["openTime"],
            y=df["ema21"],
            name="EMA 21",
            line=dict(color="#C4B5FD", width=1.3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["openTime"],
            y=df["bb_upper"],
            name="BB Upper",
            line=dict(color="#8B7BAE", width=1, dash="dot"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["openTime"],
            y=df["bb_lower"],
            name="BB Lower",
            line=dict(color="#8B7BAE", width=1, dash="dot"),
        )
    )

    fig.add_hline(
        y=live_price,
        line_color="#FDE68A",
        line_dash="dash",
        annotation_text=f"Live ${live_price:,.2f}",
    )

    fig.add_hline(
        y=analysis["support"],
        line_color="#34D399",
        line_dash="dot",
        annotation_text="Support",
    )

    fig.add_hline(
        y=analysis["resistance"],
        line_color="#FB7185",
        line_dash="dot",
        annotation_text="Resistance",
    )

    if analysis["stop"] is not None:
        fig.add_hline(
            y=analysis["stop"],
            line_color="#F87171",
            line_dash="dash",
            annotation_text="SL",
        )

    for number, target in enumerate(
        analysis["targets"], start=1
    ):
        fig.add_hline(
            y=target,
            line_color="#4ADE80",
            line_dash="dot",
            annotation_text=f"TP{number}",
        )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#17151F",
        plot_bgcolor="#17151F",
        font=dict(color="#F4F0FF"),
        height=430,
        xaxis_rangeslider_visible=False,
        margin=dict(l=8, r=8, t=45, b=8),
        legend=dict(orientation="h"),
    )

    return fig


def money(value):
    if value is None:
        return "—"
    return f"${value:,.2f}"


# -----------------------------
# SIDEBAR CONTROLS
# Keep these OUTSIDE the fragment.
# -----------------------------

with st.sidebar:
    st.subheader("Strategy settings")

    strategy = st.selectbox(
        "Signal strategy",
        [
            "Engulfing",
            "Support & Resistance",
            "Breakout",
        ],
    )

    rr = st.select_slider(
        "Risk-to-reward",
        options=[1.0, 1.5, 2.0, 2.5, 3.0],
        value=2.0,
    )

    st.caption(
        "Entry M1 · confirmation M5 · manual execution only"
    )

    st.divider()
    st.subheader("AI Smart Lot Manager")

    capital = st.number_input(
        "Account capital (USD)",
        min_value=0.0,
        value=100.0,
        step=10.0,
    )

    risk_percent = st.select_slider(
        "Risk per trade (%)",
        options=[0.5, 1.0, 1.5, 2.0],
        value=1.0,
    )

    st.caption(
        "Lot estimate is illustrative. Verify your broker's "
        "contract size and tick value."
    )


# -----------------------------
# DASHBOARD
# -----------------------------

def render_dashboard(strategy, rr, capital, risk_percent):
    if st.button("Refresh market data"):
        get_quote.clear()
        get_candles.clear()
        st.rerun()

    try:
        quote = get_quote()
        raw_m1 = get_candles("1m")
        raw_m5 = get_candles("5m")

        m1, _ = analyse(raw_m1, strategy, rr)
        m5, _ = analyse(raw_m5, strategy, rr)

        quote_price = quote.get("mid")

        if quote_price is None:
            quote_price = quote.get("bid")

        if quote_price is None:
            raise ValueError("Quote has no usable price.")

        quote_price = float(quote_price)

        stale = quote.get("stale", True)
        market_state = quote.get("marketState", "unknown")
        quote_age = quote.get("quoteAgeSeconds", "unknown")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "XAUUSD live reference",
            f"${quote_price:,.2f}",
        )

        c2.metric("M1 trend", m1["trend"])
        c3.metric("M5 trend", m5["trend"])

        st.caption(
            f"Feed: {quote.get('source', 'Unknown')} · "
            f"Quote age: {quote_age}s · "
            f"Checked {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC · "
            f"Auto-refresh every {REFRESH_SECONDS}s"
        )

        if stale or market_state != "open":
            st.warning(
                f"Feed status: {market_state}; "
                f"stale quote: {stale}. Price may be delayed."
            )

        final_signal = m1["signal"]

        if (
            final_signal == "BUY"
            and m5["trend"] != "BULLISH"
        ):
            final_signal = "WAIT"

        elif (
            final_signal == "SELL"
            and m5["trend"] != "BEARISH"
        ):
            final_signal = "WAIT"

        signal_color = {
            "BUY": "#FDE68A",
            "SELL": "#FCA5A5",
            "WAIT": "#C4B5FD",
        }[final_signal]

        st.markdown("### Current signal")

        st.markdown(
            f"""
            <div class="signal-panel">
                <div style="font-size:12px;color:#B9ADC9">
                    {strategy.upper()} · M1 SETUP / M5 CONFIRMATION
                </div>
                <div style="font-size:34px;font-weight:800;
                    color:{signal_color}">
                    {final_signal}
                </div>
                <div>{m1["reason"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        a, b, c = st.columns(3)

        if (
            final_signal != "WAIT"
            and m1["stop"] is not None
        ):
            a.metric(
                "Entry zone",
                f'{m1["entry_low"]:,.2f} – {m1["entry_high"]:,.2f}',
            )

            b.metric("Stop loss", money(m1["stop"]))

            c.metric("TP 2", money(m1["targets"][1]))

            x, y, z = st.columns(3)

            x.metric("TP 1", money(m1["targets"][0]))
            y.metric("TP 2", money(m1["targets"][1]))
            z.metric("TP 3", money(m1["targets"][2]))

            risk_cash = capital * risk_percent / 100

            stop_gap = abs(
                m1["price"] - m1["stop"]
            )

            estimated_lot = (
                risk_cash / (stop_gap * 100)
                if stop_gap > 0 else 0
            )

            st.caption(
                f"Illustrative risk budget: ${risk_cash:.2f}. "
                f"Rough lot estimate: {estimated_lot:.3f}. "
                "This assumes 1 lot = 100 oz and a $1 price move "
                "is approximately $100 per lot. Verify broker specs."
            )

        else:
            a.metric("Entry zone", "—")
            b.metric("Stop loss", "—")
            c.metric("Take profit", "—")

            st.caption(
                "No confirmed setup. Wait for the selected strategy "
                "and M5 trend to agree."
            )

        st.markdown("### Charts")

        left, right = st.columns(2)

        with left:
            st.plotly_chart(
                draw_chart(
                    raw_m1,
                    "M1 · Entry analysis",
                    quote_price,
                    m1,
                ),
                use_container_width=True,
            )

        with right:
            st.plotly_chart(
                draw_chart(
                    raw_m5,
                    "M5 · Trend confirmation",
                    quote_price,
                    m5,
                ),
                use_container_width=True,
            )

        st.markdown("### Support & resistance")

        a, b, c = st.columns(3)

        a.metric("Recent support", money(m1["support"]))
        b.metric("Recent resistance", money(m1["resistance"]))
        c.metric("M1 ATR", money(m1["atr"]))

        st.markdown("### Indicator snapshot")

        a, b, c = st.columns(3)

        a.metric("Bollinger upper", money(m1["bb_upper"]))
        b.metric("Bollinger middle", money(m1["bb_mid"]))
        c.metric("Bollinger lower", money(m1["bb_lower"]))

        st.markdown("### Signal history")

        st.caption(
            "This version displays the current analysis only. "
            "Persistent win/loss history is not enabled."
        )

        history = pd.DataFrame([{
            "UTC candle time": str(m1["time"]),
            "Symbol": "XAUUSD",
            "Strategy": strategy,
            "Signal": final_signal,
            "M1 trend": m1["trend"],
            "M5 trend": m5["trend"],
            "Live quote": round(quote_price, 2),
            "Reason": m1["reason"],
        }])

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True,
        )

        st.warning(
            "Experimental signals, not financial advice. "
            "Biquote may differ from XM/HFM; delayed quotes and "
            "false breakouts are possible. Support/resistance is "
            "approximate. The lot estimate assumes a common 100 oz "
            "contract and may not match your broker. Verify levels "
            "and test on demo."
        )

    except requests.RequestException as error:
        st.error(
            "Could not reach the market-data provider. Try again later."
        )
        st.caption(f"Technical detail: {error}")

    except (
        ValueError,
        KeyError,
        TypeError,
        IndexError,
    ) as error:
        st.error(
            "The provider returned missing or unexpected data."
        )
        st.caption(f"Technical detail: {error}")


st.title("GOLD SIGNAL AI")

st.caption(
    "XAUUSD · M1 entries · M5 confirmation · manual execution only"
)


# Automatic refresh only wraps the dashboard.
# Sidebar controls remain outside the fragment.
@st.fragment(run_every=REFRESH_SECONDS)
def dashboard_fragment():
    render_dashboard(
        strategy,
        rr,
        capital,
        risk_percent,
    )


dashboard_fragment()
