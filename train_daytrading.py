#!/usr/bin/env python3
"""
AI Stock GPT - Day Trading Model Training Pipeline.

Fetches intraday 5-minute OHLCV data from Alpha Vantage, builds
intraday features, and trains XGBoost+LightGBM ensemble models
with a short-horizon (60-minute) directional target.

Uses the same XGBoostPredictor as the daily pipeline -- only the
feature pipeline and target horizon differ.

Usage:
    python train_daytrading.py                            # Train all day-trading symbols
    python train_daytrading.py --quick                    # Quick subset (5 symbols)
    python train_daytrading.py --symbols AAPL,TSLA,SPY    # Specific symbols
    python train_daytrading.py --months 6                  # Use 6 months of data
    python train_daytrading.py --tune                      # Optuna hyperparameter tuning
    python train_daytrading.py --discover                  # Auto-discover new symbols via API
"""

import os
import sys
import time
import json
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from backend.ml.daytrading_feature_pipeline import DayTradingFeaturePipeline
from backend.ml.xgboost_model import XGBoostPredictor
from backend.data.alphavantage_collector import AlphaVantageCollector
from config import DAY_TRADING_CONFIG, DAY_TRADING_SYMBOLS, DAY_TRADING_QUICK_SYMBOLS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join("models", "daytrading")


def parse_args():
    parser = argparse.ArgumentParser(description="AI Stock GPT Day Trading Model Training")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated symbols to train")
    parser.add_argument("--quick", action="store_true",
                        help="Train only 5 symbols for quick testing")
    parser.add_argument("--discover", action="store_true",
                        help="Auto-discover new symbols from Alpha Vantage LISTING_STATUS")
    parser.add_argument("--months", type=int,
                        default=DAY_TRADING_CONFIG["months_history"],
                        help="Months of intraday history to fetch")
    parser.add_argument("--horizon", type=int,
                        default=DAY_TRADING_CONFIG["target_horizon_bars"],
                        help="Target horizon in bars (default: 12 = 60 min)")
    parser.add_argument("--interval", type=str,
                        default=DAY_TRADING_CONFIG["interval"],
                        help="Bar interval (default: 5min)")
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna hyperparameter tuning before training")
    parser.add_argument("--tune-trials", type=int, default=30,
                        help="Number of Optuna trials per model")
    parser.add_argument("--tune-timeout", type=int, default=300,
                        help="Optuna timeout in seconds")
    return parser.parse_args()


def discover_new_symbols(av_collector, models_dir):
    """Find symbols that have active listings but no trained model yet."""
    # Get existing trained models
    existing = set()
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            if f.endswith("_xgboost.joblib"):
                sym = f.replace("_xgboost.joblib", "").replace("_", "=").replace("_", "^")
                existing.add(sym.upper())

    # Also include symbols from the default training lists
    from train_models import ALL_SYMBOLS
    all_known = set(s.upper() for s in ALL_SYMBOLS) | existing

    discovered = av_collector.discover_new_symbols(
        existing_symbols=list(all_known),
        us_exchanges=("NYSE", "NASDAQ"),
        asset_types=("Stock",),
        min_days_listed=30,
    )

    new_us = discovered.get("us_new", [])
    new_india = discovered.get("india_new", [])

    logger.info(f"Auto-discovery: {len(new_us)} new US symbols, {len(new_india)} new India symbols")
    if new_us:
        logger.info(f"  New US (first 20): {new_us[:20]}")
    if new_india:
        logger.info(f"  New India: {new_india}")

    return new_us + new_india


def train_daytrading_symbols(
    symbols,
    pipeline,
    predictor,
    months,
    horizon,
    interval,
    tune=False,
    tune_trials=30,
    tune_timeout=300,
):
    """Train day-trading models for a list of symbols."""
    tuner = None
    if tune:
        try:
            from backend.ml.hyperparam_tuner import HyperparamTuner
            tuner = HyperparamTuner(params_dir=MODELS_DIR)
            logger.info("Optuna hyperparameter tuning ENABLED")
        except ImportError:
            logger.warning("Optuna not available, skipping tuning")

    min_rows = DAY_TRADING_CONFIG["min_training_rows"]
    results = {}
    total_start = time.time()

    for i, symbol in enumerate(symbols, 1):
        logger.info(f"\n[{i}/{len(symbols)}] Day-trading training: {symbol}")
        start = time.time()

        try:
            # Build intraday features
            logger.info(f"  Fetching {months} months of {interval} data...")
            features = pipeline.build_features(
                symbol, months=months, interval=interval
            )
            logger.info(f"  Intraday features shape: {features.shape}")

            # Optuna tuning (optional)
            tuned_params = None
            if tuner:
                cached = tuner.load_params(f"DT_{symbol}")
                if cached:
                    logger.info(f"  Using cached tuned params for DT_{symbol}")
                    tuned_params = cached
                else:
                    logger.info(f"  Running Optuna tuning for DT_{symbol}...")
                    target = predictor._create_target(features, horizon=horizon)
                    tuned_params = tuner.tune(
                        f"DT_{symbol}", features, target.values,
                        n_trials=tune_trials, timeout=tune_timeout,
                    )

            # Train using the same XGBoostPredictor with shorter horizon
            logger.info(f"  Training ensemble (horizon={horizon} bars)...")
            metrics = predictor.train(
                symbol, features,
                tuned_params=tuned_params,
                horizon=horizon,
                min_rows=min_rows,
            )

            elapsed = time.time() - start
            results[symbol] = {
                "status": "success",
                "accuracy": metrics["accuracy"],
                "calibrated_accuracy": metrics.get("calibrated_accuracy", metrics["accuracy"]),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "n_features": metrics.get("n_features", 0),
                "ensemble": metrics.get("ensemble", False),
                "elapsed": round(elapsed, 1),
            }

            logger.info(f"  {symbol} done in {elapsed:.1f}s")
            logger.info(f"  Accuracy: {metrics['accuracy']:.2%}  "
                        f"Calibrated: {metrics.get('calibrated_accuracy', 0):.2%}")
            logger.info(f"  Precision: {metrics['precision']:.2%}  "
                        f"Recall: {metrics['recall']:.2%}  "
                        f"F1: {metrics['f1']:.2%}")
            logger.info(f"  Features: {metrics.get('n_features', '?')}  "
                        f"Ensemble: {metrics.get('ensemble', False)}")
            logger.info(f"  Top features: "
                        f"{list(metrics.get('feature_importance', {}).keys())[:5]}")

            # Latest prediction
            prediction = predictor.predict(symbol, features)
            logger.info(f"  Signal: {prediction['direction']} "
                        f"(prob={prediction['probability']:.2%}, "
                        f"confidence={prediction['confidence']:.2%})")

        except Exception as e:
            elapsed = time.time() - start
            results[symbol] = {
                "status": "failed",
                "error": str(e),
                "elapsed": round(elapsed, 1),
            }
            logger.error(f"  FAILED: {e}")

    total_elapsed = time.time() - total_start
    return results, total_elapsed


def print_summary(results, total_elapsed, horizon):
    """Print day-trading training summary."""
    logger.info("\n" + "=" * 70)
    logger.info("DAY TRADING TRAINING SUMMARY")
    logger.info(f"Target horizon: {horizon} bars")
    logger.info("=" * 70)

    succeeded = [s for s, r in results.items() if r["status"] == "success"]
    failed = [s for s, r in results.items() if r["status"] == "failed"]

    logger.info(f"Total: {len(results)} | Succeeded: {len(succeeded)} | Failed: {len(failed)}")
    logger.info(f"Total time: {total_elapsed:.1f}s")

    if succeeded:
        import numpy as np
        accs = [results[s]["accuracy"] for s in succeeded]
        cal_accs = [results[s].get("calibrated_accuracy", results[s]["accuracy"]) for s in succeeded]
        f1s = [results[s]["f1"] for s in succeeded]
        precs = [results[s]["precision"] for s in succeeded]
        recs = [results[s]["recall"] for s in succeeded]

        logger.info(f"\nAccuracy      - mean: {np.mean(accs):.2%}  "
                    f"median: {np.median(accs):.2%}  std: {np.std(accs):.2%}")
        logger.info(f"Cal. Accuracy - mean: {np.mean(cal_accs):.2%}  "
                    f"median: {np.median(cal_accs):.2%}")
        logger.info(f"Precision     - mean: {np.mean(precs):.2%}  "
                    f"median: {np.median(precs):.2%}")
        logger.info(f"Recall        - mean: {np.mean(recs):.2%}  "
                    f"median: {np.median(recs):.2%}")
        logger.info(f"F1            - mean: {np.mean(f1s):.2%}  "
                    f"median: {np.median(f1s):.2%}")

    logger.info(f"\nPer-symbol results (sorted by accuracy):")
    sorted_results = sorted(
        [(s, r) for s, r in results.items() if r["status"] == "success"],
        key=lambda x: x[1]["accuracy"],
        reverse=True,
    )
    for symbol, r in sorted_results:
        ens = "E" if r.get("ensemble") else " "
        logger.info(
            f"  {symbol:12s}  acc={r['accuracy']:.2%}  "
            f"cal={r.get('calibrated_accuracy', 0):.2%}  "
            f"f1={r['f1']:.2%}  prec={r['precision']:.2%}  "
            f"rec={r['recall']:.2%}  feat={r.get('n_features', '?')}  "
            f"[{ens}]  ({r['elapsed']:.1f}s)"
        )

    if failed:
        logger.info(f"\nFailed symbols:")
        for symbol in failed:
            logger.info(f"  {symbol:12s}  {results[symbol].get('error', 'unknown')}")

    logger.info(f"\nModels saved to: {os.path.abspath(MODELS_DIR)}/")

    # Save report
    report_path = os.path.join(MODELS_DIR, "daytrading_report.json")
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "type": "daytrading",
        "horizon_bars": DAY_TRADING_CONFIG["target_horizon_bars"],
        "total_symbols": len(results),
        "succeeded": len(succeeded),
        "failed": len(failed),
        "total_time": total_elapsed,
        "results": results,
    }
    if succeeded:
        import numpy as np
        report["avg_accuracy"] = float(np.mean(accs))
        report["avg_calibrated_accuracy"] = float(np.mean(cal_accs))
        report["avg_f1"] = float(np.mean(f1s))
        report["median_accuracy"] = float(np.median(accs))

    try:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Report saved to: {os.path.abspath(report_path)}")
    except Exception as e:
        logger.warning(f"Failed to save report: {e}")


def main():
    args = parse_args()

    logger.info("=" * 70)
    logger.info("AI Stock GPT - Day Trading Model Training")
    logger.info("=" * 70)

    os.makedirs(MODELS_DIR, exist_ok=True)

    av_collector = AlphaVantageCollector()

    # Select symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    elif args.quick:
        symbols = DAY_TRADING_QUICK_SYMBOLS
    elif args.discover:
        # Auto-discover new symbols not yet trained
        new_syms = discover_new_symbols(av_collector, MODELS_DIR)
        # Also retrain existing day-trading symbols
        symbols = list(DAY_TRADING_SYMBOLS) + new_syms
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for s in symbols:
            if s.upper() not in seen:
                seen.add(s.upper())
                deduped.append(s)
        symbols = deduped
        logger.info(f"Discovery mode: {len(symbols)} total symbols "
                    f"({len(DAY_TRADING_SYMBOLS)} existing + {len(new_syms)} discovered)")
    else:
        symbols = DAY_TRADING_SYMBOLS
    logger.info(f"Alpha Vantage API key: "
                f"{'*' * max(0, len(av_collector.api_key) - 4)}"
                f"{av_collector.api_key[-4:] if av_collector.api_key else 'NOT SET'}")
    logger.info(f"Symbols: {len(symbols)} | Interval: {args.interval} | "
                f"Months: {args.months} | Horizon: {args.horizon} bars")

    pipeline = DayTradingFeaturePipeline(av_collector=av_collector)
    predictor = XGBoostPredictor(models_dir=MODELS_DIR)

    results, total_elapsed = train_daytrading_symbols(
        symbols,
        pipeline,
        predictor,
        months=args.months,
        horizon=args.horizon,
        interval=args.interval,
        tune=args.tune,
        tune_trials=args.tune_trials,
        tune_timeout=args.tune_timeout,
    )

    print_summary(results, total_elapsed, args.horizon)


if __name__ == "__main__":
    main()
