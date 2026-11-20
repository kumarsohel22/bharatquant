import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def make_features(candles: pd.DataFrame) -> pd.DataFrame:
    """Create causal features. No centered windows or future values are used."""
    df = candles.copy()
    o, h, l, c, v = (df[x].astype(float) for x in
                     ["open", "high", "low", "close", "volume"])

    candle_range = (h - l).replace(0, np.nan)
    df["return_1"] = c.pct_change(1)
    df["return_2"] = c.pct_change(2)
    df["return_3"] = c.pct_change(3)
    df["return_5"] = c.pct_change(5)
    df["log_return_1"] = np.log(c).diff()

    df["body_pct"] = (c - o) / o
    df["range_pct"] = (h - l) / c
    df["upper_wick_pct"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / c
    df["lower_wick_pct"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / c
    df["close_position"] = (c - l) / candle_range

    for period in [5, 10, 20, 50]:
        ema = c.ewm(span=period, adjust=False, min_periods=period).mean()
        df[f"ema_{period}_distance"] = c / ema - 1
        df[f"volatility_{period}"] = df["log_return_1"].rolling(period).std()

    df["rsi_14"] = _rsi(c, 14)

    prev_close = c.shift(1)
    true_range = pd.concat(
        [(h - l), (h - prev_close).abs(), (l - prev_close).abs()],
        axis=1
    ).max(axis=1)
    df["atr_14_pct"] = (
        true_range.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / c
    )

    vol_mean = v.rolling(20).mean()
    df["volume_ratio_20"] = v / vol_mean.replace(0, np.nan)
    df["volume_change_1"] = v.pct_change().replace([np.inf, -np.inf], np.nan)

    # Intraday VWAP resets each calendar day. Uses current and prior bars only.
    ts = pd.to_datetime(df["timestamp"])
    date_key = ts.dt.date
    typical_price = (h + l + c) / 3
    cumulative_pv = (typical_price * v).groupby(date_key).cumsum()
    cumulative_volume = v.groupby(date_key).cumsum().replace(0, np.nan)
    vwap = cumulative_pv / cumulative_volume
    df["vwap_distance"] = c / vwap - 1

    # Time-of-day cyclic features; timestamp's displayed local clock is used.
    minute_of_day = ts.dt.hour * 60 + ts.dt.minute
    df["minute_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    df["minute_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)

    return df.replace([np.inf, -np.inf], np.nan)


FEATURE_COLUMNS = [
    "return_1", "return_2", "return_3", "return_5", "log_return_1",
    "body_pct", "range_pct", "upper_wick_pct", "lower_wick_pct",
    "close_position",
    "ema_5_distance", "ema_10_distance", "ema_20_distance", "ema_50_distance",
    "volatility_5", "volatility_10", "volatility_20", "volatility_50",
    "rsi_14", "atr_14_pct", "volume_ratio_20", "volume_change_1",
    "vwap_distance", "minute_sin", "minute_cos",
]
