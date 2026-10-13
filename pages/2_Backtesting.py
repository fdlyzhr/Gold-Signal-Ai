
import time
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta, timezone

API = "https://biquote.io/api"
DAYS = 90
CHUNK_BARS = 900
RR = 2.0
MAX_HOLD_BARS = 240

st.set_page_config(
    page_title="Gold Signal AI | Backtest",
    page_icon="📊",
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
</style>
""", unsafe_allow_html=True)

st.title("GOLD SIGNAL AI")
st.subheader("Historical Backtesting")
st.caption(
    "XAUUSD · M1 entries · M5 confirmation · 90-day test · "
    "Manual trading only"
)


def fetch_chunk(interval, start, end):
    response = requests.get(
        f"{API}/XAUUSD/ohlc",
        params={
            "interval": interval,
            "from": start.isoformat().replace("+00:00", "Z"),
            "to": end.isoformat().replace("+00:00", "Z"),
            "limit": 1000,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()

    if not isinstance(payload, dict) or "bars" not in payload:
        raise ValueError(f"Unexpected {interval} response format.")

    return payload["bars"]


@st.cache_data(ttl=3600, show_spinner=False)
def download_history(days):
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=days)

    all_bars = {"1m": [], "5m": []}

    for interval, minutes in [("1m", 1), ("5m", 5)]:
        chunk_duration = timedelta(minutes=minutes * CHUNK_BARS)
        cursor = start

        while cursor < end:
            chunk_end = min(cursor + chunk_duration, end)

            bars = fetch_chunk(interval, cursor, chunk_end)
            all_bars[interval].extend(bars)

            # Move forward even if the provider returns no bars.
            cursor = chunk_end

            # Brief pause between requests.
            time.sleep(0.05)

    result = {}

    for interval, bars in all_bars.items():
        df = pd.DataFrame(bars)

        required = ["openTime", "open", "high", "low", "close"]

        if df.empty or not all(col in df.columns for col in required):
            result[interval] = pd.DataFrame()
            continue

        df["openTime"] = pd.to_datetime(
            df["openTime"], utc=True, errors="coerce"
        )

        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=required)

        if "isOpen" in df.columns:
            df = df[df["isOpen"] != True]

        df = df.drop_duplicates("openTime")
        df = df.sort_values("openTime").reset_index(drop=True)
        df = df[
            (df["openTime"] >= pd.Timestamp(start))
            & (df["openTime"] < pd.Timestamp(end))
        ]

        result[interval] = df.reset_index(drop=True)

    return result, start, end


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

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr"] = tr.rolling(14).mean()

    previous = df.shift(1)

    df["bullish_engulfing"] = (
        (previous["close"] < previous["open"])
        & (df["close"] > df["open"])
        & (df["open"] <= previous["close"])
        & (df["close"] >= previous["open"])
    )

    df["bearish_engulfing"] = (
        (previous["close"] > previous["open"])
        & (df["close"] < df["open"])
        & (df["open"] >= previous["close"])
        & (df["close"] <= previous["open"])
    )

    df["trend"] = "NEUTRAL"

    df.loc[
        (df["ema9"] > df["ema21"])
        & (df["close"] > df["ema50"]),
        "trend",
    ] = "BULLISH"

    df.loc[
        (df["ema9"] < df["ema21"])
        & (df["close"] < df["ema50"]),
        "trend",
    ] = "BEARISH"

    # These levels use historical candles available at that time.
    df["support"] = df["low"].shift(1).rolling(10).min()
    df["resistance"] = df["high"].shift(1).rolling(10).max()

    return df


def build_signals(m1, m5):
    m1 = add_indicators(m1)
    m5 = add_indicators(m5)

    # A candle becomes usable only after its closing time.
    m1["signal_time"] = m1["openTime"] + pd.Timedelta(minutes=1)
    m5["confirm_time"] = m5["openTime"] + pd.Timedelta(minutes=5)

    confirmation = m5[["confirm_time", "trend"]].rename(
        columns={"trend": "m5_trend"}
    ).sort_values("confirm_time")

    m1 = pd.merge_asof(
        m1.sort_values("signal_time"),
        confirmation,
        left_on="signal_time",
        right_on="confirm_time",
        direction="backward",
        allow_exact_matches=True,
    )

    m1["signal"] = "WAIT"

    buy = (
        (m1["trend"] == "BULLISH")
        & (m1["m5_trend"] == "BULLISH")
        & m1["bullish_engulfing"]
    )

    sell = (
        (m1["trend"] == "BEARISH")
        & (m1["m5_trend"] == "BEARISH")
        & m1["bearish_engulfing"]
    )

    m1.loc[buy, "signal"] = "BUY"
    m1.loc[sell, "signal"] = "SELL"

    return m1


def run_backtest(
    df,
    spread,
    slippage,
    max_hold,
):
    trades = []
    i = 1

    while i < len(df) - 1:
        row = df.iloc[i]

        if row["signal"] not in ("BUY", "SELL"):
            i += 1
            continue

        # Enter at the next candle's open, never at the signal candle close.
        entry_row = df.iloc[i + 1]
        direction = row["signal"]
        entry = float(entry_row["open"])

        atr = row["atr"]
        support = row["support"]
        resistance = row["resistance"]

        if pd.isna(atr) or pd.isna(support) or pd.isna(resistance):
            i += 1
            continue

        atr = float(atr)

        if direction == "BUY":
            stop = min(float(support), entry - 1.2 * atr)
            risk = entry - stop
            target = entry + RR * risk
        else:
            stop = max(float(resistance), entry + 1.2 * atr)
            risk = stop - entry
            target = entry - RR * risk

        if risk <= 0:
            i += 1
            continue

        exit_index = None
        exit_price = None
        outcome = None

        last_index = min(len(df) - 1, i + 1 + max_hold)

        for j in range(i + 1, last_index + 1):
            candle = df.iloc[j]

            if direction == "BUY":
                stop_hit = candle["low"] <= stop
                target_hit = candle["high"] >= target
            else:
                stop_hit = candle["high"] >= stop
                target_hit = candle["low"] <= target

            # Conservative assumption when both levels are touched
            # within the same candle: count the stop loss first.
            if stop_hit:
                exit_index = j
                exit_price = stop
                outcome = "LOSS"
                break

            if target_hit:
                exit_index = j
                exit_price = target
                outcome = "WIN"
                break

        if exit_index is None:
            exit_index = last_index
            exit_price = float(df.iloc[exit_index]["close"])
            outcome = "OPEN" if exit_index == len(df) - 1 else "TIME EXIT"

        gross = (
            exit_price - entry
            if direction == "BUY"
            else entry - exit_price
        )

        # Estimated cost model, in USD per ounce of gold.
        # This is not a verified HFM historical spread.
        total_cost = spread + (2 * slippage)
        net = gross - total_cost

        trades.append({
            "Entry time": entry_row["openTime"],
            "Direction": direction,
            "Entry": entry,
            "Stop loss": stop,
            "Take profit": target,
            "Exit time": df.iloc[exit_index]["openTime"],
            "Exit price": exit_price,
            "Outcome": outcome,
            "Gross move ($/oz)": gross,
            "Estimated cost ($/oz)": total_cost,
            "Net move ($/oz)": net,
            "R multiple before costs": gross / risk,
        })

        # Avoid overlapping trades in this simple first version.
        i = exit_index + 1

    return pd.DataFrame(trades)


st.markdown("### Backtest settings")

a, b, c = st.columns(3)

with a:
    st.metric("Test period", "90 days")

with b:
    spread = st.number_input(
        "Estimated spread ($/oz)",
        min_value=0.0,
        max_value=20.0,
        value=0.30,
        step=0.05,
        help=(
            "Illustrative default only. Replace with a representative "
            "HFM spread for your account and trading hours."
        ),
    )

with c:
    slippage = st.number_input(
        "Slippage per side ($/oz)",
        min_value=0.0,
        max_value=10.0,
        value=0.10,
        step=0.05,
        help="Estimated adverse price movement per entry or exit.",
    )

max_hold = st.number_input(
    "Maximum holding time (minutes)",
    min_value=5,
    max_value=1440,
    value=240,
    step=15,
)

st.warning(
    "The spread and slippage defaults are illustrative, not verified HFM "
    "costs. Results are price-movement statistics per ounce, not account "
    "profit in dollars. Actual P&L depends on lot size, contract size, "
    "commission, swaps and execution."
)

if st.button("Run 90-day backtest", type="primary"):
    try:
        with st.spinner(
            "Downloading historical M1 and M5 candles. "
            "This may take several minutes..."
        ):
            data, start, end = download_history(DAYS)

        m1 = data["1m"]
        m5 = data["5m"]

        st.write(
            f"Requested UTC period: {start:%Y-%m-%d %H:%M} to "
            f"{end:%Y-%m-%d %H:%M}"
        )

        st.write(f"Retrieved M1 candles: {len(m1):,}")
        st.write(f"Retrieved M5 candles: {len(m5):,}")

        expected_m1 = DAYS * 24 * 60
        coverage = len(m1) / expected_m1 * 100

        st.metric("M1 candle coverage", f"{min(coverage, 100):.1f}%")

        if len(m1) < 1000 or len(m5) < 200:
            st.error(
                "Insufficient historical candles for a meaningful test. "
                "The provider may not have enough history available. "
                "No performance result has been generated."
            )
            st.stop()

        signals = build_signals(m1, m5)

        trades = run_backtest(
            signals,
            spread=spread,
            slippage=slippage,
            max_hold=int(max_hold),
        )

        if trades.empty:
            st.warning(
                "No eligible setups were found in the retrieved data. "
                "This is not evidence that the strategy is profitable "
                "or unprofitable."
            )
            st.stop()

        # Exclude trades still open at the end from win-rate calculations.
        completed = trades[
            trades["Outcome"].isin(["WIN", "LOSS", "TIME EXIT"])
        ].copy()

        if completed.empty:
            st.warning("No completed trades are available to score.")
            st.stop()

        wins = (completed["Net move ($/oz)"] > 0).sum()
        losses = (completed["Net move ($/oz)"] <= 0).sum()
        win_rate = wins / len(completed) * 100

        gross_wins = completed.loc[
            completed["Net move ($/oz)"] > 0,
            "Net move ($/oz)"
        ].sum()

        gross_losses = -completed.loc[
            completed["Net move ($/oz)"] < 0,
            "Net move ($/oz)"
        ].sum()

        profit_factor = (
            gross_wins / gross_losses if gross_losses > 0 else None
        )

        cumulative = completed["Net move ($/oz)"].cumsum()
        drawdown = cumulative - cumulative.cummax()
        max_drawdown = abs(float(drawdown.min()))

        st.divider()
        st.subheader("Backtest results")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Completed trades", f"{len(completed):,}")
        k2.metric("Win rate", f"{win_rate:.1f}%")
        k3.metric(
            "Profit factor",
            f"{profit_factor:.2f}" if profit_factor is not None else "N/A",
        )
        k4.metric("Max drawdown ($/oz)", f"{max_drawdown:.2f}")

        net_total = completed["Net move ($/oz)"].sum()
        st.metric("Net cumulative move ($/oz)", f"{net_total:.2f}")

        st.caption(
            "These metrics are calculated from the retrieved data and "
            "selected cost assumptions. They are not HFM account returns. "
            "The test is a simplified historical simulation, not a "
            "guarantee of future results."
        )

        chart_data = completed.copy()
        chart_data["Trade number"] = range(1, len(chart_data) + 1)
        chart_data["Cumulative net move"] = (
            chart_data["Net move ($/oz)"].cumsum()
        )

        fig = px.line(
            chart_data,
            x="Trade number",
            y="Cumulative net move",
            title="Cumulative net price movement per ounce",
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#071B14",
            plot_bgcolor="#071B14",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Trade history")
        st.dataframe(
            completed.sort_values("Entry time", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        csv = completed.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download backtest results",
            data=csv,
            file_name="gold_signal_ai_backtest.csv",
            mime="text/csv",
        )

    except requests.RequestException as error:
        st.error(
            "The historical data provider could not be reached. "
            "Try again later. No backtest result has been fabricated."
        )
        st.caption(str(error))
    except (ValueError, KeyError, TypeError, IndexError) as error:
        st.error(
            "The provider returned missing or unexpected historical data. "
            "Check the technical detail before relying on any result."
        )
        st.caption(str(error))
