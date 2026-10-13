
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime, timezone

API = "https://biquote.io/api"
RR_RATIO = 2.0

st.set_page_config(
    page_title="Gold Signal AI",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
.stApp { background: #071B14; color: #E4F4E9; }
[data-testid="stHeader"] { background: #071B14; }
[data-testid="stSidebar"] { background: #0B241A; }
h1, h2, h3, p, label { color: #E4F4E9; }
[data-testid="stMetric"] {
    background: #103322;
    border: 1px solid #28523D;
    padding: 14px;
    border-radius: 12px;
}
[data-testid="stMetricValue"] { color: #80E0A6; }
.stButton button {
    background: #176B45;
    color: white;
    border: 1px solid #4AA879;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=15)
def get_quote():
    response = requests.get(f"{API}/XAUUSD", timeout=10)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=20)
def get_candles(interval):
    response = requests.get(
        f"{API}/XAUUSD/ohlc",
        params={"interval": interval, "limit": 300},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()

    if not isinstance(data, dict) or "bars" not in data:
        raise ValueError("The provider returned an unexpected candle format.")

    df = pd.DataFrame(data["bars"])
    required = ["openTime", "open", "high", "low", "close"]

    if df.empty or not all(col in df.columns for col in required):
        raise ValueError("Required candle fields are missing.")

    df["openTime"] = pd.to_datetime(
        df["openTime"], utc=True, errors="coerce"
    )

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required)
    df = df.sort_values("openTime").reset_index(drop=True)

    if "isOpen" in df.columns:
        df = df[df["isOpen"] != True].copy()

    if len(df) < 50:
        raise ValueError("Not enough completed candles to analyse.")

    return df.reset_index(drop=True)


def add_indicators(df):
    df = df.copy()

    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    df["bb_mid"] = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * std
    df["bb_lower"] = df["bb_mid"] - 2 * std

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


def analyse(df):
    df = add_indicators(df)
    last = df.iloc[-1]
    previous = df.iloc[-2]

    if pd.isna(last["atr"]) or pd.isna(last["bb_mid"]):
        raise ValueError("Indicators are not ready.")

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

    if last["ema9"] > last["ema21"] and last["close"] > last["ema50"]:
        trend = "BULLISH"
    elif last["ema9"] < last["ema21"] and last["close"] < last["ema50"]:
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"

    # Recent price range for approximate support/resistance and Fibonacci.
    swing = df.tail(100)
    swing_high = float(swing["high"].max())
    swing_low = float(swing["low"].min())
    swing_range = swing_high - swing_low

    if swing_range <= 0:
        raise ValueError("Price range is not usable.")

    fib_382 = swing_high - swing_range * 0.382
    fib_500 = swing_high - swing_range * 0.500
    fib_618 = swing_high - swing_range * 0.618

    # Check whether price is near the 38.2%-61.8% retracement area.
    fib_low = min(fib_382, fib_618)
    fib_high = max(fib_382, fib_618)
    near_fib = fib_low <= last["close"] <= fib_high

    # Bollinger Bands are used as context, not as an automatic reversal rule.
    near_lower_band = last["close"] <= last["bb_mid"]
    near_upper_band = last["close"] >= last["bb_mid"]

    buy_score = 0
    sell_score = 0
    buy_reasons = []
    sell_reasons = []

    if trend == "BULLISH":
        buy_score += 2
        buy_reasons.append("M1 trend is bullish")
    if trend == "BEARISH":
        sell_score += 2
        sell_reasons.append("M1 trend is bearish")

    if bullish_engulfing:
        buy_score += 2
        buy_reasons.append("Bullish engulfing candle")
    if bearish_engulfing:
        sell_score += 2
        sell_reasons.append("Bearish engulfing candle")

    if near_fib:
        buy_score += 1
        sell_score += 1

    if near_lower_band:
        buy_score += 1
        buy_reasons.append("Price is below the Bollinger middle band")
    if near_upper_band:
        sell_score += 1
        sell_reasons.append("Price is above the Bollinger middle band")

    signal = "WAIT"
    reasons = ["Conditions for a balanced setup are not fully met."]

    # Require a trend and a candle confirmation.
    if buy_score >= 4 and bullish_engulfing and trend == "BULLISH":
        signal = "BUY"
        reasons = buy_reasons
    elif sell_score >= 4 and bearish_engulfing and trend == "BEARISH":
        signal = "SELL"
        reasons = sell_reasons

    price = float(last["close"])
    atr = float(last["atr"])

    # Use recent swing points for stop placement, with an ATR safety distance.
    recent = df.tail(10)
    support = float(recent["low"].min())
    resistance = float(recent["high"].max())

    stop_loss = None
    take_profit = None
    risk_distance = None

    if signal == "BUY":
        stop_loss = min(support, price - atr * 1.2)
        risk_distance = price - stop_loss
        take_profit = price + RR_RATIO * risk_distance
    elif signal == "SELL":
        stop_loss = max(resistance, price + atr * 1.2)
        risk_distance = stop_loss - price
        take_profit = price - RR_RATIO * risk_distance

    if signal != "WAIT" and (risk_distance is None or risk_distance <= 0):
        signal = "WAIT"
        reasons = ["Invalid risk distance; wait for another setup."]
        stop_loss = None
        take_profit = None

    return {
        "signal": signal,
        "trend": trend,
        "price": price,
        "support": support,
        "resistance": resistance,
        "swing_high": swing_high,
        "swing_low": swing_low,
        "fib_382": fib_382,
        "fib_500": fib_500,
        "fib_618": fib_618,
        "bb_upper": float(last["bb_upper"]),
        "bb_mid": float(last["bb_mid"]),
        "bb_lower": float(last["bb_lower"]),
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "reasons": reasons,
        "time": last["openTime"],
    }, df


def draw_chart(df, title):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df["openTime"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="XAUUSD",
    ))

    fig.add_trace(go.Scatter(
        x=df["openTime"], y=df["ema9"],
        name="EMA 9", line=dict(width=1),
    ))
    fig.add_trace(go.Scatter(
        x=df["openTime"], y=df["ema21"],
        name="EMA 21", line=dict(width=1),
    ))
    fig.add_trace(go.Scatter(
        x=df["openTime"], y=df["bb_upper"],
        name="BB Upper", line=dict(width=1, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=df["openTime"], y=df["bb_lower"],
        name="BB Lower", line=dict(width=1, dash="dot"),
    ))

    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#071B14",
        plot_bgcolor="#071B14",
        height=460,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(orientation="h"),
    )
    return fig


st.title("GOLD SIGNAL AI")
st.caption("XAUUSD · M1 entry · M5 confirmation · Manual execution only")

with st.sidebar:
    st.subheader("Strategy settings")
    st.write("Style: Balanced")
    st.write("Entry timeframe: M1")
    st.write("Confirmation: M5")
    st.write("Target: 1:2 risk-to-reward")
    st.write("Execution: Manual only")
    st.caption("No orders are sent to your broker.")

if st.button("Refresh market data"):
    st.cache_data.clear()
    st.rerun()

try:
    quote = get_quote()
    raw_m1 = get_candles("1m")
    raw_m5 = get_candles("5m")

    m1, df_m1 = analyse(raw_m1)
    m5, df_m5 = analyse(raw_m5)

    quote_price = quote.get("mid")
    if quote_price is None:
        quote_price = quote.get("bid")

    if quote_price is None:
        raise ValueError("The quote has no usable price.")

    quote_price = float(quote_price)
    stale = quote.get("stale", True)
    market_state = quote.get("marketState", "unknown")
    quote_age = quote.get("quoteAgeSeconds", "unknown")

    if stale or market_state != "open":
        st.warning(
            f"Feed status: {market_state}. Stale quote: {stale}. "
            "The displayed price may be delayed."
        )

    st.caption(
        f"Provider: {quote.get('source', 'Unknown')} · "
        f"Quote age: {quote_age} seconds · "
        f"Last checked UTC: {datetime.now(timezone.utc).strftime('%H:%M:%S')}"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("XAUUSD live reference", f"${quote_price:,.2f}")
    c2.metric("M1 trend", m1["trend"])
    c3.metric("M5 trend", m5["trend"])

    final_signal = m1["signal"]
    if final_signal == "BUY" and m5["trend"] != "BULLISH":
        final_signal = "WAIT"
    elif final_signal == "SELL" and m5["trend"] != "BEARISH":
        final_signal = "WAIT"

    st.divider()
    st.subheader("Trade signal")

    color = {
        "BUY": "#80E0A6",
        "SELL": "#FF8D8D",
        "WAIT": "#E4F4E9",
    }[final_signal]

    st.markdown(
        f"<h1 style='color:{color};'>{final_signal}</h1>",
        unsafe_allow_html=True,
    )

    if final_signal == "WAIT":
        st.write(
            "No confirmed setup. M1 entry conditions and M5 trend "
            "must agree before a potential trade is displayed."
        )
    else:
        st.write("M1 setup and M5 trend agree.")
        a, b, c = st.columns(3)
        a.metric("Reference entry", f"${m1['price']:,.2f}")
        b.metric("Stop loss", f"${m1['stop_loss']:,.2f}")
        c.metric("Take profit (1:2)", f"${m1['take_profit']:,.2f}")

        st.caption(
            "These are reference levels calculated from completed candles. "
            "They are not broker quotes or guaranteed execution prices."
        )

    st.subheader("Why this signal?")
    for reason in m1["reasons"]:
        st.write(f"- {reason}")
    st.write(f"- M5 confirmation: {m5['trend']}")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(draw_chart(df_m1, "M1 — Entry analysis"),
                        use_container_width=True)
    with right:
        st.plotly_chart(draw_chart(df_m5, "M5 — Trend confirmation"),
                        use_container_width=True)

    st.subheader("Price levels")
    a, b = st.columns(2)
    a.metric("Recent support", f"${m1['support']:,.2f}")
    b.metric("Recent resistance", f"${m1['resistance']:,.2f}")

    st.write("**Fibonacci retracement — recent 100-candle range**")
    f1, f2, f3 = st.columns(3)
    f1.metric("38.2%", f"${m1['fib_382']:,.2f}")
    f2.metric("50.0%", f"${m1['fib_500']:,.2f}")
    f3.metric("61.8%", f"${m1['fib_618']:,.2f}")

    st.write("**Bollinger Bands — M1**")
    f1, f2, f3 = st.columns(3)
    f1.metric("Upper band", f"${m1['bb_upper']:,.2f}")
    f2.metric("Middle band", f"${m1['bb_mid']:,.2f}")
    f3.metric("Lower band", f"${m1['bb_lower']:,.2f}")

    st.info(
        "Risk warning: indicators can lag and false signals occur. "
        "The feed may differ from XM or HFM. Test signals on a demo "
        "account and account for spread, slippage and trading costs. "
        "A 1:2 target does not mean the strategy will be profitable."
    )

except requests.RequestException as error:
    st.error("Could not reach the market-data provider. Try again later.")
    st.caption(f"Technical detail: {error}")
except (ValueError, KeyError, TypeError, IndexError) as error:
    st.error("The provider returned missing or unexpected data.")
    st.caption(f"Technical detail: {error}")
