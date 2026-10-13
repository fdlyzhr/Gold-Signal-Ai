
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime, timezone

API = "https://biquote.io/api"

st.set_page_config(
    page_title="Gold Signal AI",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #071B14;
    color: #E4F4E9;
}
[data-testid="stHeader"] {
    background: #071B14;
}
[data-testid="stSidebar"] {
    background: #0B241A;
}
h1, h2, h3, p, label {
    color: #E4F4E9;
}
div[data-testid="stMetric"] {
    background: #103322;
    border: 1px solid #28523D;
    padding: 16px;
    border-radius: 12px;
}
div[data-testid="stMetricValue"] {
    color: #80E0A6;
}
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
    response = requests.get(
        f"{API}/XAUUSD",
        timeout=10
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=30)
def get_candles(interval, limit=250):
    response = requests.get(
        f"{API}/XAUUSD/ohlc",
        params={"interval": interval, "limit": limit},
        timeout=15
    )
    response.raise_for_status()
    data = response.json()

    if not isinstance(data, dict) or "bars" not in data:
        raise ValueError("Unexpected candle data format.")

    df = pd.DataFrame(data["bars"])

    required = ["openTime", "open", "high", "low", "close"]
    if df.empty or not all(c in df.columns for c in required):
        raise ValueError("No usable candle data was returned.")

    df["openTime"] = pd.to_datetime(
        df["openTime"], utc=True, errors="coerce"
    )

    for column in ["open", "high", "low", "close"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=required)
    df = df.sort_values("openTime").reset_index(drop=True)

    # Use completed candles only to avoid signals changing mid-candle.
    if "isOpen" in df.columns:
        df = df[df["isOpen"] != True].copy()

    if len(df) < 30:
        raise ValueError("Not enough completed candles for analysis.")

    return df


def analyse(df):
    df = df.copy()

    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["sma20"] = df["close"].rolling(20).mean()
    df["std20"] = df["close"].rolling(20).std()

    df["upper"] = df["sma20"] + 2 * df["std20"]
    df["lower"] = df["sma20"] - 2 * df["std20"]

    previous = df.iloc[-2]
    last = df.iloc[-1]

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

    if last["ema9"] > last["ema21"]:
        trend = "BULLISH"
    elif last["ema9"] < last["ema21"]:
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"

    # Approximate nearby support/resistance from recent completed candles.
    recent = df.tail(20)
    support = recent["low"].min()
    resistance = recent["high"].max()

    reasons = []
    signal = "WAIT"

    # Require agreement between EMA direction and candle confirmation.
    if trend == "BULLISH" and bullish_engulfing:
        signal = "BUY"
        reasons.append("EMA trend is bullish")
        reasons.append("Bullish engulfing candle confirmed")
    elif trend == "BEARISH" and bearish_engulfing:
        signal = "SELL"
        reasons.append("EMA trend is bearish")
        reasons.append("Bearish engulfing candle confirmed")
    else:
        reasons.append("Trend and candle confirmation do not agree")

    price = float(last["close"])

    # Simple illustrative risk levels, not guaranteed trading levels.
    recent_ranges = (
        df["high"] - df["low"]
    ).tail(14)

    atr_proxy = float(recent_ranges.mean())
    stop_distance = max(atr_proxy * 1.5, price * 0.0005)

    if signal == "BUY":
        stop_loss = price - stop_distance
        take_profit = price + stop_distance * 1.5
    elif signal == "SELL":
        stop_loss = price + stop_distance
        take_profit = price - stop_distance * 1.5
    else:
        stop_loss = None
        take_profit = None

    return {
        "signal": signal,
        "trend": trend,
        "price": price,
        "support": float(support),
        "resistance": float(resistance),
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "reasons": reasons,
        "time": last["openTime"],
        "upper": float(last["upper"]),
        "lower": float(last["lower"]),
        "sma20": float(last["sma20"]),
    }, df


st.title("GOLD SIGNAL AI")
st.caption("XAUUSD · Automated analysis · Manual execution only")

with st.sidebar:
    st.subheader("Dashboard settings")
    st.write("Market: Gold / US Dollar")
    st.write("Entry timeframe: M1")
    st.write("Confirmation timeframe: M5")
    st.write("Signal logic: EMA + engulfing candles")
    st.caption("Signals are educational analysis, not financial advice.")

if st.button("Refresh market data"):
    st.cache_data.clear()
    st.rerun()

try:
    quote = get_quote()
    m1_raw = get_candles("1m")
    m5_raw = get_candles("5m")

    m1, m1_df = analyse(m1_raw)
    m5, m5_df = analyse(m5_raw)

    quote_price = quote.get("mid")
    if quote_price is None:
        quote_price = quote.get("bid")

    if quote_price is None:
        st.error("The provider did not return a usable gold price.")
        st.stop()

    quote_price = float(quote_price)
    stale = quote.get("stale", True)
    market_state = quote.get("marketState", "unknown")
    quote_age = quote.get("quoteAgeSeconds", "unknown")

    if stale or market_state != "open":
        st.warning(
            f"Feed status: {market_state}. "
            f"Stale quote: {stale}. "
            "Treat the displayed price and signals cautiously."
        )

    st.caption(
        f"Feed source: {quote.get('source', 'Not provided')} · "
        f"Quote age: {quote_age} seconds · "
        f"UTC: {datetime.now(timezone.utc).strftime('%H:%M:%S')}"
    )

    a, b, c = st.columns(3)
    a.metric("XAUUSD price", f"${quote_price:,.2f}")
    b.metric("M1 direction", m1["trend"])
    c.metric("M5 direction", m5["trend"])

    st.divider()
    st.subheader("Current signal")

    # M5 must agree with the M1 setup before displaying a trade signal.
    final_signal = m1["signal"]
    if final_signal == "BUY" and m5["trend"] != "BULLISH":
        final_signal = "WAIT"
    elif final_signal == "SELL" and m5["trend"] != "BEARISH":
        final_signal = "WAIT"

    signal_color = {
        "BUY": "#80E0A6",
        "SELL": "#FF8D8D",
        "WAIT": "#E4F4E9"
    }[final_signal]

    st.markdown(
        f"<h2 style='color:{signal_color};'>{final_signal}</h2>",
        unsafe_allow_html=True
    )

    if final_signal == "WAIT":
        st.write(
            "No confirmed setup. Wait for M1 entry conditions "
            "and M5 trend to agree."
        )
    else:
        st.write("M1 entry conditions and M5 trend are aligned.")
        st.metric("Reference entry", f"${m1['price']:,.2f}")
        st.metric("Illustrative stop loss", f"${m1['stop_loss']:,.2f}")
        st.metric("Illustrative take profit", f"${m1['take_profit']:,.2f}")

    st.caption(
        "Entry, SL and TP are rule-based estimates from historical candles. "
        "They are not guaranteed levels and may differ from your broker's feed."
    )

    left, right = st.columns(2)

    with left:
        st.subheader("M1 price chart")
        fig = go.Figure(data=[go.Candlestick(
            x=m1_df["openTime"],
            open=m1_df["open"],
            high=m1_df["high"],
            low=m1_df["low"],
            close=m1_df["close"],
            name="XAUUSD"
        )])
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#071B14",
            plot_bgcolor="#071B14",
            height=430,
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=20, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("M5 price chart")
        fig5 = go.Figure(data=[go.Candlestick(
            x=m5_df["openTime"],
            open=m5_df["open"],
            high=m5_df["high"],
            low=m5_df["low"],
            close=m5_df["close"],
            name="XAUUSD"
        )])
        fig5.update_layout(
            template="plotly_dark",
            paper_bgcolor="#071B14",
            plot_bgcolor="#071B14",
            height=430,
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=20, b=10)
        )
        st.plotly_chart(fig5, use_container_width=True)

    st.subheader("Technical levels")
    x, y = st.columns(2)
    x.metric("Approx. support", f"${m1['support']:,.2f}")
    y.metric("Approx. resistance", f"${m1['resistance']:,.2f}")

    st.subheader("Signal explanation")
    for reason in m1["reasons"]:
        st.write(f"- {reason}")
    st.write(f"- M5 confirmation: {m5['trend']}")

except requests.RequestException as error:
    st.error(
        "Could not reach the market-data provider. "
        "Check your internet connection or try again later."
    )
    st.caption(f"Technical detail: {error}")
except (ValueError, KeyError, TypeError, IndexError) as error:
    st.error("The market feed returned unexpected or insufficient data.")
    st.caption(f"Technical detail: {error}")
