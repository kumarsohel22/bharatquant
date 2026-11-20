# BharatQuant OHLCV ML Engine (v1)

One-minute OHLCV binary classification: predict whether the next candle's close
will be above the current candle's close. Trains LightGBM and XGBoost and evaluates
them with expanding-window walk-forward validation.

## Important
- Features are causal: they use only current/past candles.
- The target uses the next candle and is shifted by one row.
- This is a research prototype, not a trading signal or profit guarantee.
- Yahoo Finance one-minute history is limited. Use it to test the pipeline, then
  ingest validated historical candles from your chosen source.
- Use one symbol per training run in v1. Do not concatenate unrelated symbols
  without adding symbol-aware validation and normalization.

## Expected input columns
Required: `timestamp, open, high, low, close, volume`
Optional: `symbol, exchange, oi, source`
Timestamps should identify the start/end convention consistently. The pipeline
sorts timestamps and removes duplicates. Store timestamps in Asia/Kolkata when
possible; timezone-aware timestamps are retained.

## Setup (Windows PowerShell)
```powershell
conda create -n bharatquant python=3.11 -y
conda activate bharatquant
cd D:\STUDY\PYTHON\bharatquant\BharatQuant
# Extract this project here, or copy its files into your existing project.
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Download a small test dataset
```powershell
python scripts\download_yfinance.py --symbol RELIANCE.NS --period 7d
```
This writes `data\raw\RELIANCE_NS_1m.parquet`.

## Train both models
```powershell
python -m src.bharatquant.train --input data\raw\RELIANCE_NS_1m.parquet --symbol RELIANCE.NS --folds 5
```

For CSV:
```powershell
python -m src.bharatquant.train --input data\raw\my_candles.csv --symbol RELIANCE.NS --folds 5
```

Outputs:
- `artifacts/<SYMBOL>/lightgbm.joblib`
- `artifacts/<SYMBOL>/xgboost.joblib`
- `artifacts/<SYMBOL>/metrics.csv`
- `artifacts/<SYMBOL>/oof_predictions.csv`
- `artifacts/<SYMBOL>/feature_columns.json`

## Prediction
After training:
```powershell
python -m src.bharatquant.predict --input data\raw\RELIANCE_NS_1m.parquet --symbol RELIANCE.NS --model lightgbm
```

## Target definition
For row t:
`target_up = 1 if close[t+1] > close[t], else 0`
Therefore, exactly unchanged next closes are currently labelled DOWN (0). Inspect
the frequency of unchanged closes; a later version can use a three-class target
(UP / DOWN / NO-TRADE) or a minimum-move threshold after costs are considered.

## Validation notes
`TimeSeriesSplit` uses expanding training windows and a one-row gap. The final
rows that lack a next candle are dropped. Metrics are classification metrics,
not trading returns. Before live use, add spread, brokerage, taxes, slippage,
order-fill assumptions, and a paper-trading period.
