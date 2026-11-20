import argparse
import re
from pathlib import Path

import pandas as pd
import yfinance as yf


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value)


def normalize_yfinance_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flatten yfinance's single- or multi-level column layout."""
    if isinstance(df.columns, pd.MultiIndex):
        selected = None
        for level in range(df.columns.nlevels):
            values = df.columns.get_level_values(level)
            if ticker in values:
                selected = df.xs(ticker, axis=1, level=level)
                break
        if selected is None:
            # Fall back to the first ticker group if only one ticker was requested.
            selected = df.copy()
            selected.columns = [
                next((str(x) for x in col if str(x).lower() in
                      {"open", "high", "low", "close", "volume", "adj close"}),
                     str(col[0]))
                for col in selected.columns
            ]
        df = selected

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    needed = ["open", "high", "low", "close", "volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns from Yahoo response: {missing}; got {list(df.columns)}")
    return df[needed].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="RELIANCE.NS", help="Yahoo ticker, e.g. RELIANCE.NS")
    parser.add_argument("--period", default="7d", help="Yahoo period, e.g. 5d or 7d")
    parser.add_argument("--outdir", default="data/raw")
    args = parser.parse_args()

    print(f"Downloading {args.symbol}, interval=1m, period={args.period}")
    raw = yf.download(
        args.symbol,
        period=args.period,
        interval="1m",
        auto_adjust=False,
        prepost=False,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError(
            "Yahoo Finance returned no rows. Check the ticker, market session, "
            "internet connection, and Yahoo's current intraday-history limits."
        )

    df = normalize_yfinance_columns(raw, args.symbol)
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        # Yahoo normally supplies a timezone-aware index. If not, assume India time
        # for NSE tickers; verify this assumption for other exchanges.
        idx = idx.tz_localize("Asia/Kolkata")
    else:
        idx = idx.tz_convert("Asia/Kolkata")

    df.insert(0, "timestamp", idx)
    df["symbol"] = args.symbol
    df["exchange"] = "NSE" if args.symbol.upper().endswith(".NS") else "UNKNOWN"
    df["source"] = "yfinance"
    df = df.reset_index(drop=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / f"{safe_name(args.symbol)}_1m.parquet"
    df.to_parquet(output, index=False)

    print(f"Saved {len(df):,} candles to: {output}")
    print(df.head(5).to_string(index=False))
    print("\nColumns:", ", ".join(df.columns))


if __name__ == "__main__":
    main()
