import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss, brier_score_loss
)
from sklearn.model_selection import TimeSeriesSplit

from .data import load_candles
from .features import make_features, FEATURE_COLUMNS


def make_dataset(candles: pd.DataFrame):
    df = make_features(candles)
    # At time t, predict whether close[t+1] is greater than close[t].
    next_close = df["close"].shift(-1)
    df["target_up"] = (next_close > df["close"]).astype("int8")
    df.loc[next_close.isna(), "target_up"] = np.nan

    usable = df.dropna(subset=FEATURE_COLUMNS + ["target_up"]).copy()
    usable["target_up"] = usable["target_up"].astype("int8")
    return usable


def build_models():
    return {
        "lightgbm": LGBMClassifier(
            objective="binary",
            n_estimators=350,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        ),
        "xgboost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=350,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        ),
    }


def calculate_metrics(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision_up": precision_score(y_true, predictions, zero_division=0),
        "recall_up": recall_score(y_true, predictions, zero_division=0),
        "f1_up": f1_score(y_true, predictions, zero_division=0),
        "log_loss": log_loss(y_true, np.clip(probabilities, 1e-6, 1 - 1e-6), labels=[0, 1]),
        "brier_score": brier_score_loss(y_true, probabilities),
    }
    metrics["roc_auc"] = (
        roc_auc_score(y_true, probabilities) if len(np.unique(y_true)) == 2 else np.nan
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV or Parquet OHLCV file")
    parser.add_argument("--symbol", default="UNKNOWN")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--outdir", default="artifacts")
    args = parser.parse_args()

    candles = load_candles(args.input)
    data = make_dataset(candles)

    if len(data) < 500:
        raise ValueError(
            f"Only {len(data)} usable rows. Collect more candles; ideally several "
            "thousand or more for an initial experiment."
        )
    if data["target_up"].nunique() < 2:
        raise ValueError("Target contains only one class; cannot train a binary classifier.")

    X = data[FEATURE_COLUMNS]
    y = data["target_up"]
    n_splits = min(args.folds, max(2, len(data) // 200))
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=1)

    outdir = Path(args.outdir) / args.symbol.replace("/", "_").replace(".", "_")
    outdir.mkdir(parents=True, exist_ok=True)

    fold_metrics = []
    oof_parts = []
    for model_name, prototype in build_models().items():
        print(f"\n========== {model_name.upper()} walk-forward evaluation ==========")
        model_predictions = []
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
            model = build_models()[model_name]
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            if y_train.nunique() < 2 or y_test.nunique() < 2:
                print(f"Fold {fold}: skipped because train/test has only one class.")
                continue

            model.fit(X_train, y_train)
            probabilities = model.predict_proba(X_test)[:, 1]
            metrics = calculate_metrics(y_test, probabilities)
            metrics.update({
                "model": model_name,
                "fold": fold,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "test_start": str(data.iloc[test_idx[0]]["timestamp"]),
                "test_end": str(data.iloc[test_idx[-1]]["timestamp"]),
            })
            fold_metrics.append(metrics)
            print(
                f"Fold {fold}: accuracy={metrics['accuracy']:.4f}, "
                f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
                f"ROC-AUC={metrics['roc_auc']:.4f}"
            )

            part = pd.DataFrame({
                "timestamp": data.iloc[test_idx]["timestamp"].values,
                "actual_up": y_test.values,
                "probability_up": probabilities,
                "predicted_up": (probabilities >= 0.5).astype(int),
                "model": model_name,
                "fold": fold,
            })
            model_predictions.append(part)

        if model_predictions:
            oof_parts.extend(model_predictions)

        # Fit final deployable model on all currently available labelled rows.
        final_model = build_models()[model_name]
        final_model.fit(X, y)
        joblib.dump(final_model, outdir / f"{model_name}.joblib")
        print(f"Saved final {model_name} model.")

    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.to_csv(outdir / "metrics.csv", index=False)
    if oof_parts:
        pd.concat(oof_parts, ignore_index=True).to_csv(
            outdir / "oof_predictions.csv", index=False
        )

    with open(outdir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    print("\n========== SUMMARY (mean across evaluated folds) ==========")
    if not metrics_df.empty:
        print(metrics_df.groupby("model")[[
            "accuracy", "balanced_accuracy", "precision_up",
            "recall_up", "f1_up", "roc_auc", "log_loss", "brier_score"
        ]].mean().round(4).to_string())
    print(f"\nArtifacts saved in: {outdir.resolve()}")
    print("\nReminder: classification metrics are not trading returns.")


if __name__ == "__main__":
    main()
