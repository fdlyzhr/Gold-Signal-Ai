import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Gold Signal AI", page_icon="📈", layout="wide")

REQUIRED = ["open", "high", "low", "close"]

def normalize_csv(uploaded):
    """Read MT5-exported OHLC CSV and normalize common column names."""
    raw = pd.read_csv(uploaded)
    raw.columns = [str(c).strip().lower().replace("<", "").replace(">", "") for c in raw.columns]
    aliases = {
        "date": "date", "time": "time", "datetime": "datetime",
        "open": "open", "high": "high", "low": "low", "close": "close",
        "tickvol": "tick_volume", "tick_volume": "tick_volume",
        "volume": "tick_volume", "spread": "spread", "real_volume": "real_volume"
    }
    raw = raw.rename(columns={c: aliases[c] for c in raw.columns if c in aliases})

    # MT5 exports may contain separate DATE and TIME columns.
    if "datetime" not in raw.columns:
        if "date" in raw.columns and "time" in raw.columns:
            raw["datetime"] = pd.to_datetime(
                raw["date"].astype(str) + " " + raw["time"].astype(str), errors="coerce"
            )
        elif "date" in raw.columns:
            raw["datetime"] = pd.to_datetime(raw["date"], errors="coerce")
        elif "time" in raw.columns:
            raw["datetime"] = pd.to_datetime(raw["time"], errors="coerce")
        else:
            raw["datetime"] = pd.RangeIndex(start=0, stop=len(raw), step=1)

    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(
            "CSV must contain Open, High, Low and Close columns. "
            f"Missing: {', '.join(missing)}. Check the MT5 export format."
        )

    for col in REQUIRED:
        raw[col] = pd.to_numeric(raw[col].astype(str).str.replace(",", ""), errors="coerce")
    raw = raw.dropna(subset=REQUIRED).copy()
    raw = raw.sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)
    if len(raw) < 100:
        raise ValueError("Please upload at least 100 valid candles for indicator warm-up and testing.")
    return raw

def add_indicators(df, bb_period=20, bb_std=2.0, swing_period=50, sr_period=30, atr_period=14):
    d = df.copy()
    d["bb_mid"] = d["close"].rolling(bb_period).mean()
    sd = d["close"].rolling(bb_period).std(ddof=0)
    d["bb_upper"] = d["bb_mid"] + bb_std * sd
    d["bb_lower"] = d["bb_mid"] - bb_std * sd

    prev_close = d["close"].shift(1)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - prev_close).abs(),
        (d["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    d["atr"] = tr.rolling(atr_period).mean()

    # Trailing levels are shifted by one bar to avoid using the current candle's extremes.
    d["support"] = d["low"].rolling(sr_period).min().shift(1)
    d["resistance"] = d["high"].rolling(sr_period).max().shift(1)
    d["swing_low"] = d["low"].rolling(swing_period).min().shift(1)
    d["swing_high"] = d["high"].rolling(swing_period).max().shift(1)
    span = d["swing_high"] - d["swing_low"]
    d["fib_382_bull"] = d["swing_high"] - 0.382 * span
    d["fib_618_bull"] = d["swing_high"] - 0.618 * span
    d["fib_382_bear"] = d["swing_low"] + 0.382 * span
    d["fib_618_bear"] = d["swing_low"] + 0.618 * span

    # Body-engulfing definitions; both candles must be fully formed.
    po, pc = d["open"].shift(1), d["close"].shift(1)
    d["bull_engulf"] = (
        (pc < po) & (d["close"] > d["open"]) &
        (d["open"] <= pc) & (d["close"] >= po)
    )
    d["bear_engulf"] = (
        (pc > po) & (d["close"] < d["open"]) &
        (d["open"] >= pc) & (d["close"] <= po)
    )

    # Require a confluence zone, not just one indicator.
    tol = d["atr"] * 0.35
    d["near_support"] = (d["close"] - d["support"]).abs() <= tol
    d["near_resistance"] = (d["close"] - d["resistance"]).abs() <= tol
    d["bull_fib_zone"] = (
        (d["close"] <= d["fib_382_bull"]) &
        (d["close"] >= d["fib_618_bull"])
    )
    d["bear_fib_zone"] = (
        (d["close"] >= d["fib_382_bear"]) &
        (d["close"] <= d["fib_618_bear"])
    )
    d["bull_bb_ok"] = d["low"] <= d["bb_lower"] + tol
    d["bear_bb_ok"] = d["high"] >= d["bb_upper"] - tol

    # 3/4 confluence checks plus the engulfing trigger. This is a hypothesis to test,
    # not a claim that these rules are profitable.
    d["buy_signal"] = (
        d["bull_engulf"] & d["near_support"] & d["bull_fib_zone"] & d["bull_bb_ok"]
    )
    d["sell_signal"] = (
        d["bear_engulf"] & d["near_resistance"] & d["bear_fib_zone"] & d["bear_bb_ok"]
    )
    return d

def add_m5_confirmation(lower_df, m5_df):
    """Apply the last fully closed M5 Bollinger-midline bias to lower-timeframe candles.
    Assumes MT5 CSV timestamps are candle-open times.
    """
    low = lower_df.copy()
    high = m5_df.copy()
    high = add_indicators(high)
    high["m5_completed_at"] = pd.to_datetime(high["datetime"]) + pd.Timedelta(minutes=5)
    high["m5_bias"] = np.where(
        high["close"] > high["bb_mid"], "BUY",
        np.where(high["close"] < high["bb_mid"], "SELL", "NEUTRAL")
    )
    high = high[["m5_completed_at", "m5_bias"]].dropna().sort_values("m5_completed_at")
    low["_signal_time"] = pd.to_datetime(low["datetime"]) + pd.Timedelta(minutes=1)
    low = pd.merge_asof(
        low.sort_values("_signal_time"),
        high,
        left_on="_signal_time",
        right_on="m5_completed_at",
        direction="backward"
    )
    low["buy_signal"] = low["buy_signal"] & (low["m5_bias"] == "BUY")
    low["sell_signal"] = low["sell_signal"] & (low["m5_bias"] == "SELL")
    return low.drop(columns=["_signal_time", "m5_completed_at"], errors="ignore")


def create_trade_levels(row, side, rr=1.5, atr_buffer=0.2):
    entry = float(row["close"])
    atr = float(row["atr"]) if pd.notna(row["atr"]) else np.nan
    if not np.isfinite(atr) or atr <= 0:
        return None
    if side == "BUY":
        structural = min(float(row["low"]), float(row["support"])) if pd.notna(row["support"]) else float(row["low"])
        sl = structural - atr_buffer * atr
        risk = entry - sl
        if risk <= 0:
            return None
        tp = entry + rr * risk
    else:
        structural = max(float(row["high"]), float(row["resistance"])) if pd.notna(row["resistance"]) else float(row["high"])
        sl = structural + atr_buffer * atr
        risk = sl - entry
        if risk <= 0:
            return None
        tp = entry - rr * risk
    return {"side": side, "entry": entry, "sl": sl, "tp": tp, "risk_distance": risk, "rr": rr}

def backtest(d, rr=1.5, atr_buffer=0.2, max_hold=50, spread_price=0.0):
    """Simple candle backtest: signal closes, entry at next candle open.
    If SL and TP are both touched in one candle, count SL first (conservative).
    spread_price is a price-unit round-trip cost approximation, not a broker quote.
    """
    trades = []
    i = 0
    while i < len(d) - 2:
        row = d.iloc[i]
        side = "BUY" if bool(row.get("buy_signal", False)) else (
            "SELL" if bool(row.get("sell_signal", False)) else None
        )
        if not side:
            i += 1
            continue
        levels = create_trade_levels(row, side, rr, atr_buffer)
        if not levels:
            i += 1
            continue
        entry_i = i + 1
        entry = float(d.iloc[entry_i]["open"])
        # Rebase target and stop around the actual next-open fill, retaining stop distance.
        risk_dist = levels["risk_distance"]
        if side == "BUY":
            sl, tp = entry - risk_dist, entry + rr * risk_dist
        else:
            sl, tp = entry + risk_dist, entry - rr * risk_dist

        exit_i = min(len(d) - 1, entry_i + max_hold)
        outcome, exit_price, actual_exit_i = "TIME", float(d.iloc[exit_i]["close"]), exit_i
        for j in range(entry_i, exit_i + 1):
            bar = d.iloc[j]
            if side == "BUY":
                hit_sl = float(bar["low"]) <= sl
                hit_tp = float(bar["high"]) >= tp
            else:
                hit_sl = float(bar["high"]) >= sl
                hit_tp = float(bar["low"]) <= tp
            if hit_sl:  # conservative if both levels are touched in the same bar
                outcome, exit_price, actual_exit_i = "LOSS", sl, j
                break
            if hit_tp:
                outcome, exit_price, actual_exit_i = "WIN", tp, j
                break

        gross = (exit_price - entry) if side == "BUY" else (entry - exit_price)
        net = gross - max(0.0, spread_price)
        risk = abs(entry - sl)
        r_multiple = net / risk if risk > 0 else 0.0
        trades.append({
            "signal_time": row["datetime"],
            "entry_time": d.iloc[entry_i]["datetime"],
            "exit_time": d.iloc[actual_exit_i]["datetime"],
            "side": side, "entry": entry, "sl": sl, "tp": tp,
            "exit": exit_price, "result": outcome, "net_price_pnl": net,
            "R": r_multiple
        })
        i = actual_exit_i + 1  # one position at a time; no overlapping trades
    return pd.DataFrame(trades)

def fmt_price(value):
    return f"{value:,.3f}"

st.title("Gold Signal AI")
st.caption("XAUUSD technical setup scanner · M1 / M5 · CSV-based prototype")
st.warning(
    "Research prototype only. It does not connect to live prices or place orders. "
    "Signals are not financial advice and profitability is not established."
)

with st.sidebar:
    st.header("Strategy settings")
    timeframe = st.selectbox("Chart timeframe", ["M1", "M5"], index=0)
    rr = st.slider("Target reward:risk", 1.0, 3.0, 1.5, 0.25)
    atr_buffer = st.slider("Stop buffer (ATR multiple)", 0.0, 1.0, 0.2, 0.1)
    risk_pct = st.slider("Risk per trade (% of account)", 0.25, 2.0, 0.5, 0.25)
    account_size = st.number_input("Demo account balance (account currency)", min_value=0.0, value=1000.0, step=100.0)
    spread_price = st.number_input(
        "Estimated round-trip cost in XAUUSD price units",
        min_value=0.0, value=0.20, step=0.05,
        help="A rough sensitivity input only. Replace with measured broker spread, commission and slippage."
    )
    max_hold = st.slider("Maximum bars held in backtest", 5, 200, 50, 5)

st.markdown("### 1. Upload historical candles")
st.write(
    "Export XAUUSD M1 and M5 candles from MT5 as CSV. For M1 signals, upload both files to apply "
    "a closed-candle M5 Bollinger-midline trend filter. If only one file is available, the app can still "
    "analyse that timeframe without multi-timeframe confirmation. The app never needs your MT5 password."
)
c_up1, c_up2 = st.columns(2)
with c_up1:
    m1_upload = st.file_uploader("M1 CSV (lower-timeframe entries)", type=["csv"], key="m1")
with c_up2:
    m5_upload = st.file_uploader("M5 CSV (trend confirmation)", type=["csv"], key="m5")

if m1_upload is None and m5_upload is None:
    st.info("Upload an MT5 CSV to calculate signals and backtest the rules.")
    st.markdown("**Expected columns:** `Date`, `Time`, `Open`, `High`, `Low`, `Close`. A single `Datetime` column is also accepted.")
    st.markdown("**Suggested test:** export several weeks or months of candles. Results depend on broker symbol, feed, spread and time zone.")
    st.stop()

try:
    m1_data = normalize_csv(m1_upload) if m1_upload is not None else None
    m5_data = normalize_csv(m5_upload) if m5_upload is not None else None
    if timeframe == "M1" and m1_data is not None:
        source = m1_data
    elif timeframe == "M5" and m5_data is not None:
        source = m5_data
    elif m1_data is not None:
        source = m1_data
        timeframe = "M1"
    else:
        source = m5_data
        timeframe = "M5"
    d = add_indicators(source)
    if timeframe == "M1" and m5_data is not None:
        d = add_m5_confirmation(d, m5_data)
        m5_status = "M5 trend filter applied using completed M5 candles."
    elif timeframe == "M1":
        m5_status = "M5 file not supplied; multi-timeframe confirmation is OFF."
    else:
        m5_status = "M5 chart selected; M5 is the analysis timeframe."
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.caption(m5_status)

if len(d) < 100:
    st.error("Not enough candles after cleaning.")
    st.stop()

latest = d.iloc[-1]
signal = "BUY" if bool(latest.get("buy_signal", False)) else (
    "SELL" if bool(latest.get("sell_signal", False)) else "NO TRADE"
)
levels = create_trade_levels(latest, signal, rr, atr_buffer) if signal != "NO TRADE" else None

st.markdown("### 2. Latest closed-candle scan")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest close", fmt_price(float(latest["close"])))
c2.metric("Timeframe", timeframe)
c3.metric("Signal", signal)
c4.metric("Candles loaded", f"{len(d):,}")

if levels:
    a, b, c, e = st.columns(4)
    a.metric("Indicative entry", fmt_price(levels["entry"]))
    b.metric("Stop loss", fmt_price(levels["sl"]))
    c.metric("Take profit", fmt_price(levels["tp"]))
    e.metric("Reward:risk", f"1:{rr:.2f}")
    risk_cash = account_size * risk_pct / 100
    st.info(
        f"At {risk_pct:.2f}% account risk, the cash-risk budget is {risk_cash:,.2f} "
        f"(account currency). This app does not convert that budget into lot size because XAUUSD "
        "contract size, tick value and volume rules vary by broker."
    )
else:
    st.write("No qualifying setup on the latest candle. This is a valid outcome; do not force a trade.")

# Plot recent candles and BBs
plot_n = min(250, len(d))
chart = d.tail(plot_n)
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=chart["datetime"], open=chart["open"], high=chart["high"],
    low=chart["low"], close=chart["close"], name="XAUUSD"
))
fig.add_trace(go.Scatter(x=chart["datetime"], y=chart["bb_upper"], name="BB Upper", line=dict(width=1)))
fig.add_trace(go.Scatter(x=chart["datetime"], y=chart["bb_mid"], name="BB Middle", line=dict(width=1)))
fig.add_trace(go.Scatter(x=chart["datetime"], y=chart["bb_lower"], name="BB Lower", line=dict(width=1)))
if levels:
    fig.add_hline(y=levels["entry"], line_dash="dot", annotation_text="Entry")
    fig.add_hline(y=levels["sl"], line_dash="dash", annotation_text="SL")
    fig.add_hline(y=levels["tp"], line_dash="dash", annotation_text="TP")
fig.update_layout(height=560, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=25, b=10))
st.plotly_chart(fig, use_container_width=True)

st.markdown("### 3. Backtest")
st.caption(
    "The backtest enters at the next candle open after a signal candle. It assumes the stop is hit first "
    "when TP and SL are both touched within one candle. Cost is a simple price-unit deduction; validate "
    "it against your broker's actual spread, commissions and slippage."
)
trades = backtest(d, rr=rr, atr_buffer=atr_buffer, max_hold=max_hold, spread_price=spread_price)
if trades.empty:
    st.warning("No historical trades were generated by these strict confluence rules. Try a longer data sample; do not loosen rules solely to create more trades.")
else:
    wins = int((trades["R"] > 0).sum())
    losses = int((trades["R"] <= 0).sum())
    win_rate = wins / len(trades) * 100
    gross_win = trades.loc[trades["R"] > 0, "R"].sum()
    gross_loss = abs(trades.loc[trades["R"] < 0, "R"].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else np.inf
    equity = trades["R"].cumsum()
    drawdown = equity - equity.cummax()
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0
    expectancy = float(trades["R"].mean())
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Trades", f"{len(trades)}")
    m2.metric("Win rate", f"{win_rate:.1f}%")
    m3.metric("Profit factor", "∞" if np.isinf(pf) else f"{pf:.2f}")
    m4.metric("Expectancy", f"{expectancy:.2f} R/trade")
    m5.metric("Max drawdown", f"{max_dd:.2f} R")
    st.line_chart(equity.rename("Cumulative R"))
    st.dataframe(trades.tail(100), use_container_width=True, hide_index=True)
    st.download_button(
        "Download backtest trades CSV",
        data=trades.to_csv(index=False).encode("utf-8"),
        file_name="gold_signal_backtest_trades.csv",
        mime="text/csv"
    )
    st.caption("These are in-sample results until you test on a separate unseen period. They are not evidence of future profitability.")

st.markdown("### Signal rule notes")
st.markdown(
    "- **BUY:** bullish engulfing + price near trailing support + inside the trailing Fibonacci retracement zone + contact with/near the lower Bollinger Band.\n"
    "- **SELL:** bearish engulfing + price near trailing resistance + inside the opposite Fibonacci zone + contact with/near the upper Bollinger Band.\n"
    "- Support, resistance and swing levels are calculated from prior candles to reduce look-ahead bias.\n"
    "- These are deliberately strict starter rules. They may produce very few signals and need testing, not assumptions."
)
st.caption("Never enter a live trade solely because this prototype prints BUY or SELL. Test on demo first.")
