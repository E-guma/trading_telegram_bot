"""Data feed, feature engineering, walk-forward backtest and prediction
for Deriv's Volatility 100 (1s) Index (symbol 1HZ100V)."""

import json
import math

import numpy as np
import pandas as pd
import pandas_ta as ta
import websockets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

WS_URL = "wss://api.derivws.com/trading/v1/options/ws/public"
SYMBOL = "1HZ100V"
HORIZON = 5


# ---------------------------------------------------------------- data feed
async def fetch_candles(count: int = 2000, granularity: int = 900) -> pd.DataFrame:
    req = {
        "ticks_history": SYMBOL,
        "adjust_start_time": 1,
        "count": count,
        "end": "latest",
        "style": "candles",
        "granularity": granularity,
    }

    async with websockets.connect(WS_URL) as ws:
        await ws.send(json.dumps(req))
        resp = json.loads(await ws.recv())

    if "error" in resp:
        raise RuntimeError(resp["error"]["message"])

    df = pd.DataFrame(resp.get("candles", []))
    df["time"] = pd.to_datetime(df["epoch"], unit="s")
    df = df.set_index("time")[["open", "high", "low", "close"]].astype(float)

    return df


# ----------------------------------------------------------------- features
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, o = df["close"], df["high"], df["low"], df["open"]
    f = pd.DataFrame(index=df.index)

    # 1. Essential Momentum & Volatility (Trimmed Correlated Noise)
    f["rsi"] = ta.rsi(c, length=14)
    f["atr_pct"] = ta.atr(h, l, c, length=14) / c

    # 2. Core Triple EMA System
    ema9 = ta.ema(c, length=9)
    ema26 = ta.ema(c, length=26)
    ema50 = ta.ema(c, length=50)
    f["ema9_26_gap"] = (ema9 - ema26) / c

    # 3. Vectorized Geometric Wick Math
    body = (c - o).abs().clip(lower=1e-5)
    bull_wick = ((np.minimum(o, c) - l) >= (body * 1.8)).astype(int)
    bear_wick = ((h - np.maximum(o, c)) >= (body * 1.8)).astype(int)

    # 4. EMA & Static S/R Rejections (Combined for UI Prices)
    ema_tol, static_tol = 0.005, 0.001
    rolling_max, rolling_min = h.rolling(70).max(), l.rolling(70).min()

    # Calculate individual rejection triggers for both 26 and 50 EMAs
    bull_rej_26 = ((l <= ema26 * (1 + ema_tol)) & (c > ema26) & bull_wick).astype(int)
    bull_rej_50 = ((l <= ema50 * (1 + ema_tol)) & (c > ema50) & bull_wick).astype(int)
    bear_rej_26 = ((h >= ema26 * (1 - ema_tol)) & (c < ema26) & bear_wick).astype(int)
    bear_rej_50 = ((h >= ema50 * (1 - ema_tol)) & (c < ema50) & bear_wick).astype(int)

    f["bull_ema_rej"] = np.maximum(bull_rej_26, bull_rej_50)
    f["bear_ema_rej"] = np.maximum(bear_rej_26, bear_rej_50)

    f["bull_static_rej"] = ((l <= rolling_min * (1 + static_tol)) & bull_wick).astype(
        int
    )
    f["bear_static_rej"] = ((h >= rolling_max * (1 - static_tol)) & bear_wick).astype(
        int
    )

    # UI Only: Capture exact rejection prices (Quarantined from ML later)
    f["bull_rej_price"] = np.where(f["bull_ema_rej"] | f["bull_static_rej"], l, np.nan)
    f["bear_rej_price"] = np.where(f["bear_ema_rej"] | f["bear_static_rej"], h, np.nan)

    f["recent_bull_rej"] = f["bull_ema_rej"].rolling(window=3).max().fillna(0)
    f["recent_bear_rej"] = f["bear_ema_rej"].rolling(window=3).max().fillna(0)

    f["recent_bull_price"] = f["bull_rej_price"].ffill(limit=2).fillna(0)
    f["recent_bear_price"] = f["bear_rej_price"].ffill(limit=2).fillna(0)
    f = f.drop(columns=["bull_rej_price", "bear_rej_price"])

    # 5. Market Regime & Exhaustion
    adx = ta.adx(h, l, c, length=14)
    f["adx"] = adx.iloc[:, 0] if adx is not None else 0
    f["is_ranging"] = (f["adx"] < 20).astype(int)

    bb = ta.bbands(c, length=20, std=2)
    bb_lower = bb.iloc[:, 0] if bb is not None else c
    bb_upper = bb.iloc[:, 2] if bb is not None else c
    f["bull_exhaustion"] = (l < bb_lower).astype(int)
    f["bear_exhaustion"] = (h > bb_upper).astype(int)

    # 6. Dynamic Confluence Scoring System
    ema_rej_weight = 2 - f["is_ranging"]
    static_rej_weight = 1 + f["is_ranging"]

    f["bull_score"] = (
        (ema9 > ema26).astype(int)
        + (c > ema50).astype(int)
        + (f["recent_bull_rej"] * ema_rej_weight)
        + (f["bull_static_rej"] * static_rej_weight)
        + f["bull_exhaustion"]
        + ((f["rsi"] > 40) & (f["rsi"] < 70)).astype(int)
    )

    f["bear_score"] = (
        (ema9 < ema26).astype(int)
        + (c < ema50).astype(int)
        + (f["recent_bear_rej"] * ema_rej_weight)
        + (f["bear_static_rej"] * static_rej_weight)
        + f["bear_exhaustion"]
        + ((f["rsi"] < 60) & (f["rsi"] > 30)).astype(int)
    )

    return f


def _model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1))


def _dataset(df: pd.DataFrame):
    X_all = add_features(df)

    # Quarantine UI prices to prevent model drift
    ml_cols = [
        col
        for col in X_all.columns
        if col not in ["recent_bull_price", "recent_bear_price"]
    ]
    X = X_all[ml_cols]

    y = (df["close"].shift(-HORIZON) > df["close"]).astype(int)
    valid = X.notna().all(axis=1) & df["close"].shift(-HORIZON).notna()
    return X[valid], y[valid], X_all


# ----------------------------------------------------------------- backtest
def walk_forward(df: pd.DataFrame, n_splits: int = 5) -> dict:
    X, y, _ = _dataset(df)
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=HORIZON)
    correct, total, baseline_hits, folds = 0, 0, 0, []

    for tr, te in tscv.split(X):
        m = _model().fit(X.iloc[tr], y.iloc[tr])
        pred = m.predict(X.iloc[te])
        hits = int((pred == y.iloc[te].values).sum())
        majority = int(y.iloc[tr].mode()[0])
        baseline_hits += int((y.iloc[te].values == majority).sum())
        correct += hits
        total += len(te)
        folds.append(hits / len(te))

    acc = correct / total
    ci = 1.96 * math.sqrt(acc * (1 - acc) / total)
    return {
        "accuracy": acc,
        "ci": ci,
        "baseline": baseline_hits / total,
        "folds": folds,
        "n_test": total,
    }


# --------------------------------------------------------------- prediction
def analyse(df: pd.DataFrame, tf_label: str = "15-Min") -> dict:
    X, y, X_all = _dataset(df)
    model = _model().fit(X, y)

    latest_all = X_all.iloc[[-1]]
    ml_cols = [
        c for c in X_all.columns if c not in ["recent_bull_price", "recent_bear_price"]
    ]
    latest_ml = latest_all[ml_cols]

    p_up = float(model.predict_proba(latest_ml)[0, 1])
    row = latest_all.iloc[0]

    bull_score = int(row["bull_score"])
    bear_score = int(row["bear_score"])

    # Determine highest confluence state
    if bull_score >= 3:
        confluence_grade = "HIGH" if bull_score >= 4 else "MODERATE"
        confluence_dir = "BULLISH"
        confluence_score = bull_score
    elif bear_score >= 3:
        confluence_grade = "HIGH" if bear_score >= 4 else "MODERATE"
        confluence_dir = "BEARISH"
        confluence_score = bear_score
    else:
        confluence_grade = "LOW"
        confluence_dir = "NEUTRAL"
        confluence_score = max(bull_score, bear_score)

    has_bull_rej = bool(row["recent_bull_rej"] or row["bull_static_rej"])
    has_bear_rej = bool(row["recent_bear_rej"] or row["bear_static_rej"])

    rej_price = None
    if has_bull_rej:
        rej_price = float(row["recent_bull_price"])
    elif has_bear_rej:
        rej_price = float(row["recent_bear_price"])

    adx_val = float(row["adx"])
    regime = "RANGING" if adx_val < 20 else "TRENDING"

    if bool(row["bull_exhaustion"]):
        exhaustion_state = "Downside Exhaustion (Oversold bounds breached)"
    elif bool(row["bear_exhaustion"]):
        exhaustion_state = "Upside Exhaustion (Overbought bounds breached)"
    else:
        exhaustion_state = "Normal bounds"

    return {
        "symbol": SYMBOL,
        "asset_name": "Volatility 100 (1s) Index",
        "timeframe": tf_label,
        "price": float(df["close"].iloc[-1]),
        "p_up": p_up,
        "confluence_dir": confluence_dir,
        "confluence_grade": confluence_grade,
        "confluence_score": confluence_score,
        "ema_trend": "up" if row["ema9_26_gap"] > 0 else "down",
        "has_rejection": has_bull_rej or has_bear_rej,
        "rej_price": rej_price,
        "rsi": float(row["rsi"]),
        "atr": float(row["atr_pct"]) * float(df["close"].iloc[-1]),
        "market_regime": regime,
        "adx": adx_val,
        "exhaustion_state": exhaustion_state,
    }


def format_report(a: dict, bt: dict | None = None) -> str:
    direction = "UP" if a["p_up"] >= 0.5 else "DOWN"
    conf = max(a["p_up"], 1 - a["p_up"])

    header_icon = (
        "🟢"
        if a["confluence_dir"] == "BULLISH"
        else "🔴"
        if a["confluence_dir"] == "BEARISH"
        else "🟡"
    )

    if a["has_rejection"]:
        price_text = f" at {a['rej_price']:.2f}" if a["rej_price"] else ""
        rej_text = f"Structural rejection wick confirmed{price_text}"
    else:
        rej_text = "No major structural rejection"

    lines = [
        f"<b>{header_icon} V100 (1s) | {a.get('timeframe', '15-Min')} Outlook: {a['confluence_dir']}</b>",
        f"Price: <code>{a['price']:.2f}</code>\n",
        f"🎯 <b>Setup Quality:</b> {a['confluence_grade']} PROBABILITY (Grade {a['confluence_score']}/7+)",
        f"• Fast Cross: 9/26 EMA {'Bullish' if a['ema_trend'] == 'up' else 'Bearish'}",
        f"• Price Action: {rej_text}",
        f"• Exhaustion: {a['exhaustion_state']}",
        f"• Model Lean: <b>{direction}</b> (<code>{conf:.0%}</code> ML probability)\n",
        "🛡️ <b>Market Context</b>",
        f"• Regime: {a['market_regime']} (ADX: <code>{a['adx']:.1f}</code>)",
        f"• Volatility: ATR <code>{a['atr']:.2f}</code>",
        f"• Momentum: RSI <code>{a['rsi']:.1f}</code>",
    ]

    if bt:
        edge = bt["accuracy"] - 0.5
        lines.append(
            f"\n<i>Out-of-sample accuracy: {bt['accuracy']:.1%} ± {bt['ci']:.1%} "
            f"(baseline {bt['baseline']:.1%}, n={bt['n_test']})</i>"
        )
        if abs(edge) <= bt["ci"]:
            lines.append("⚠️ No statistically meaningful edge. Treat the lean as noise.")

    lines.append("\n<i>Not financial advice. This index is RNG-generated.</i>")
    return "\n".join(lines)
