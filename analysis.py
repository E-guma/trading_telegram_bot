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
HORIZON = 3 


# ---------------------------------------------------------------- data feed
async def fetch_candles(count: int = 2000, granularity: int = 300) -> pd.DataFrame:
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
    
    df = pd.DataFrame(resp["candles"])
    df["time"] = pd.to_datetime(df["epoch"], unit="s")
    df = df.set_index("time")[["open", "high", "low", "close"]].astype(float)
    return df


# ----------------------------------------------------------------- features
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]
    f = pd.DataFrame(index=df.index)
    
    # 1. Base returns & momentum indicators
    for k in (1, 2, 3, 6, 12):
        f[f"ret{k}"] = c.pct_change(k)
    f["rsi"] = ta.rsi(c, length=14)
    macd = ta.macd(c, fast=12, slow=26, signal=9)
    f["macd_hist"] = macd["MACDh_12_26_9"] / c
    f["atr_pct"] = ta.atr(h, l, c, length=14) / c
    r = c.pct_change()
    f["vol_ratio"] = ta.stdev(r, length=10) / ta.stdev(r, length=60)

    # 2. Triple EMA System (9, 26, 50)
    ema9 = ta.ema(c, length=9)
    ema26 = ta.ema(c, length=26)
    ema50 = ta.ema(c, length=50)

    f["ema9_26_gap"] = (ema9 - ema26) / c
    f["ema26_50_gap"] = (ema26 - ema50) / c
    f["price_ema50_gap"] = (c - ema50) / c

    # 3. Geometric Wick Math
    body = (c - o).abs().clip(lower=1e-5)
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l

    bull_wick = (lower_wick >= (body * 1.8)).astype(int)
    bear_wick = (upper_wick >= (body * 1.8)).astype(int)

    # 4. Proximity Rejection of 26 or 50 EMA (0.1% buffer zone)
    tolerance = 0.001
    
    bull_rej_26 = ((l <= ema26 * (1 + tolerance)) & (c > ema26) & (bull_wick == 1)).astype(int)
    bull_rej_50 = ((l <= ema50 * (1 + tolerance)) & (c > ema50) & (bull_wick == 1)).astype(int)

    bear_rej_26 = ((h >= ema26 * (1 - tolerance)) & (c < ema26) & (bear_wick == 1)).astype(int)
    bear_rej_50 = ((h >= ema50 * (1 - tolerance)) & (c < ema50) & (bear_wick == 1)).astype(int)

    f["bull_ema_rej"] = np.maximum(bull_rej_26, bull_rej_50)
    f["bear_ema_rej"] = np.maximum(bear_rej_26, bear_rej_50)

    # 5. Confluence Scoring System
    f["bull_score"] = (
        (ema9 > ema26).astype(int)
        + (c > ema50).astype(int)
        + (f["bull_ema_rej"] * 2)
        + ((f["rsi"] > 40) & (f["rsi"] < 70)).astype(int)
    )

    f["bear_score"] = (
        (ema9 < ema26).astype(int)
        + (c < ema50).astype(int)
        + (f["bear_ema_rej"] * 2)
        + ((f["rsi"] < 60) & (f["rsi"] > 30)).astype(int)
    )

    return f


def _model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.1))


def _dataset(df: pd.DataFrame):
    X = add_features(df)
    y = (df["close"].shift(-HORIZON) > df["close"]).astype(int)
    valid = X.notna().all(axis=1) & df["close"].shift(-HORIZON).notna()
    return X[valid], y[valid], X


# ----------------------------------------------------------------- backtest
def walk_forward(df: pd.DataFrame, n_splits: int = 5) -> dict:
    """Out-of-sample test. `gap` stops overlapping labels leaking into the test set."""
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
    ci = 1.96 * math.sqrt(acc * (1 - acc) / total)  # optimistic: ignores overlap
    return {
        "accuracy": acc,
        "ci": ci,
        "baseline": baseline_hits / total,
        "folds": folds,
        "n_test": total,
    }


# --------------------------------------------------------------- prediction
def analyse(df: pd.DataFrame) -> dict:
    X, y, X_all = _dataset(df)
    model = _model().fit(X, y)
    latest = X_all.iloc[[-1]]
    p_up = float(model.predict_proba(latest)[0, 1])
    row = latest.iloc[0]
    
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

    atr_val = float(row["atr_pct"]) * float(df["close"].iloc[-1])
    vol_regime = "high" if row["vol_ratio"] > 1.3 else "low" if row["vol_ratio"] < 0.7 else "normal"

    return {
        "price": float(df["close"].iloc[-1]),
        "p_up": p_up,
        "confluence_dir": confluence_dir,
        "confluence_grade": confluence_grade,
        "confluence_score": confluence_score,
        "ema_trend": "up" if row["ema9_26_gap"] > 0 else "down",
        "has_rejection": bool(row["bull_ema_rej"] or row["bear_ema_rej"]),
        "rsi": float(row["rsi"]),
        "macd_hist": float(row["macd_hist"]),
        "atr": atr_val,
        "vol_regime": vol_regime,
    }


def format_report(a: dict, bt: dict | None = None) -> str:
    direction = "UP" if a["p_up"] >= 0.5 else "DOWN"
    conf = max(a["p_up"], 1 - a["p_up"])
    
    header_icon = "🟢" if a["confluence_dir"] == "BULLISH" else "🔴" if a["confluence_dir"] == "BEARISH" else "🟡"
    
    lines = [
        f"<b>{header_icon} V100 (1s) | 15-Min Outlook: {a['confluence_dir']}</b>",
        f"Price: <code>{a['price']:.2f}</code>\n",
        f"🎯 <b>Setup Quality:</b> {a['confluence_grade']} PROBABILITY (Grade {a['confluence_score']}/4+)",
        f"• Fast Cross: 9/26 EMA {'Bullish' if a['ema_trend'] == 'up' else 'Bearish'}",
        f"• Price Action: {'Rejection wick printed off value zone' if a['has_rejection'] else 'No major structural rejection'}",
        f"• Model Lean: <b>{direction}</b> (<code>{conf:.0%}</code> ML probability)\n",
        "🛡️ <b>Market Context</b>",
        f"• Volatility: {a['vol_regime'].title()} (ATR: <code>{a['atr']:.2f}</code>)",
        f"• Momentum: RSI <code>{a['rsi']:.1f}</code> | MACD <code>{a['macd_hist']:+.5f}</code>"
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
