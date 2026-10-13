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

WS_URL = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
SYMBOL = "1HZ100V"
HORIZON = 15  # minutes (1-minute candles)


# ---------------------------------------------------------------- data feed
async def fetch_candles(count: int = 5000, granularity: int = 60) -> pd.DataFrame:
    req = {
        "ticks_history": SYMBOL,
        "adjust_start_time": 1,
        "count": count,
        "end": "latest",
        "style": "candles",
        "granularity": granularity,
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    async with websockets.connect(WS_URL, extra_headers=headers) as ws:
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
    f = pd.DataFrame(index=df.index)
    for k in (1, 3, 5, 15, 30):
        f[f"ret{k}"] = c.pct_change(k)
    f["rsi"] = ta.rsi(c, length=14)
    macd = ta.macd(c, fast=12, slow=26, signal=9)
    f["macd_hist"] = macd["MACDh_12_26_9"] / c
    f["ema_gap"] = (ta.ema(c, length=20) - ta.ema(c, length=50)) / c
    f["atr_pct"] = ta.atr(df["high"], df["low"], c, length=14) / c
    r = c.pct_change()
    f["vol_ratio"] = ta.stdev(r, length=10) / ta.stdev(r, length=60)
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
    atr_pct = float(row["atr_pct"])
    vol_regime = "high" if row["vol_ratio"] > 1.3 else "low" if row["vol_ratio"] < 0.7 else "normal"
    return {
        "price": float(df["close"].iloc[-1]),
        "p_up": p_up,
        "rsi": float(row["rsi"]),
        "macd_hist": float(row["macd_hist"]),
        "ema_trend": "up" if row["ema_gap"] > 0 else "down",
        "atr": atr_pct * float(df["close"].iloc[-1]),
        "vol_regime": vol_regime,
    }


def format_report(a: dict, bt: dict | None = None) -> str:
    direction = "UP" if a["p_up"] >= 0.5 else "DOWN"
    conf = max(a["p_up"], 1 - a["p_up"])
    lines = [
        "V100 (1s) | 15-min outlook",
        f"Price: {a['price']:.2f}",
        f"Model lean: {direction} ({conf:.0%} model probability)",
        f"EMA20/50 trend: {a['ema_trend']}",
        f"RSI(14): {a['rsi']:.1f}",
        f"MACD hist: {a['macd_hist']:+.5f}",
        f"ATR(14, 1m): {a['atr']:.2f} | vol regime: {a['vol_regime']}",
    ]
    if bt:
        edge = bt["accuracy"] - 0.5
        lines.append(
            f"\nOut-of-sample accuracy: {bt['accuracy']:.1%} ± {bt['ci']:.1%} "
            f"(baseline {bt['baseline']:.1%}, n={bt['n_test']})"
        )
        if abs(edge) <= bt["ci"]:
            lines.append("No statistically meaningful edge. Treat the lean as noise.")
    lines.append("\nNot financial advice. This index is RNG-generated.")
    return "\n".join(lines)
