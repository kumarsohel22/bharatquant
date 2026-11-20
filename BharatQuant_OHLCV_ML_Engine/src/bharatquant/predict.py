import argparse
from pathlib import Path

import joblib
import pandas as pd

from .data import load_candles
from .features import make_features, FEATURE_COLUMNS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--symbol", default="UNKNOWN")
    parser.add_argument("--model", choices=["lightgbm", "xgboost"], default="lightgbm")
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()

    model_dir = Path(args.artifacts) / args.symbol.replace("/", "_").replace(".", "_")
    model_path = model_dir / f"{args.model}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train first.")

    candles = load_candles(args.input)
    features = make_features(candles).dropna(subset=FEATURE_COLUMNS)
    if features.empty:
        raise ValueError("Not enough candles to calculate all features.")

    model = joblib.load(model_path)
    latest = features.iloc[[-1]]
    probability_up = float(model.predict_proba(latest[FEATURE_COLUMNS])[:, 1][0])
    direction = "UP" if probability_up >= 0.5 else "DOWN"

    print(f"Symbol: {args.symbol}")
    print(f"Last candle timestamp: {latest.iloc[0]['timestamp']}")
    print(f"Last candle close: {latest.iloc[0]['close']:.2f}")
    print(f"Predicted next-close direction: {direction}")
    print(f"Probability UP: {probability_up:.2%}")
    print(f"Probability DOWN: {1 - probability_up:.2%}")
    print("This is a model estimate, not a guaranteed outcome or trade instruction.")


if __name__ == "__main__":
    main()
