from pathlib import Path
import pandas as pd

REQUIRED = ["timestamp", "open", "high", "low", "close", "volume"]


def load_candles(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Input must be .csv, .parquet, or .pq")

    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Required schema: {REQUIRED}"
        )

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=REQUIRED)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    # Basic OHLC integrity checks.
    invalid = (
        (df["high"] < df[["open", "close", "low"]].max(axis=1)) |
        (df["low"] > df[["open", "close", "high"]].min(axis=1)) |
        (df[["open", "high", "low", "close"]] <= 0).any(axis=1) |
        (df["volume"] < 0)
    )
    if invalid.any():
        print(f"Warning: dropping {int(invalid.sum())} rows with invalid OHLCV values.")
        df = df.loc[~invalid].reset_index(drop=True)

    return df
