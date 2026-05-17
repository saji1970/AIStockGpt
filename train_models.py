#!/usr/bin/env python3
"""
AI Stock GPT - Enhanced Model Training with Optuna Tuning & Benchmarking.

Modes:
  --train          Train all models (default)
  --tune           Run Optuna hyperparameter tuning before training
  --benchmark      Compare v1 vs v2 accuracy side-by-side
  --symbols AAPL,MSFT  Train only specific symbols
  --quick          Train a small subset (10 symbols) for quick testing
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

from backend.ml.feature_pipeline import FeaturePipeline
from backend.ml.xgboost_model import XGBoostPredictor
from backend.data.alphavantage_collector import AlphaVantageCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── Full symbol list ──────────────────────────────────────────
# Tech & Growth
TECH = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
    'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'IBM', 'CSCO', 'QCOM',
    'AVGO', 'TXN', 'MU', 'AMAT', 'KLAC', 'LRCX', 'ADI', 'MCHP',
]
# Finance & Payments
FINANCE = ['PYPL', 'SQ', 'V', 'MA', 'JPM', 'BAC', 'WFC', 'GS', 'MS']
# Consumer & Retail
CONSUMER = ['DIS', 'NKE', 'SBUX', 'MCD', 'KO', 'PEP', 'WMT', 'TGT', 'HD', 'LOW', 'COST', 'PG']
# Healthcare & Pharma
HEALTH = ['TMO', 'ABBV', 'JNJ', 'PFE', 'MRK', 'UNH', 'CVS', 'CI', 'HUM', 'ELV', 'DHR']
# Industrials & Defense
INDUSTRIAL = ['GE', 'BA', 'CAT', 'DE', 'MMM', 'HON', 'RTX', 'LMT', 'NOC', 'GD', 'LHX']
# Energy
ENERGY = ['XOM', 'CVX']
# Index ETFs
INDEX_ETFS = ['SPY', 'QQQ', 'DIA', 'IWM', 'VOO', 'VTI']
# Sector ETFs
SECTOR_ETFS = ['XLK', 'XLV', 'XLF', 'XLE', 'XLY', 'VGT']
# Bond ETFs
BOND_ETFS = ['BND', 'AGG', 'TLT']
# Dividend & Income ETFs
DIVIDEND_ETFS = ['VIG', 'SCHD', 'VYM']
# Hedge Fund Alternative ETFs
HEDGE_FUND_ALTS = ['BTAL', 'DBMF', 'QMOM', 'MNA']
# Real Estate & Commodity
REAL_ESTATE_COMMODITY = ['VNQ', 'GLD']
# International
INTERNATIONAL = ['VXUS']

# ── India (BSE-listed, Nifty 50 / Sensex constituents) ───────
INDIA_TECH = ['INFY.BSE', 'TCS.BSE', 'WIPRO.BSE', 'HCLTECH.BSE', 'TECHM.BSE']
INDIA_FINANCE = [
    'HDFCBANK.BSE', 'ICICIBANK.BSE', 'SBIN.BSE', 'KOTAKBANK.BSE',
    'AXISBANK.BSE', 'BAJFINANCE.BSE', 'BAJAJFINSV.BSE', 'INDUSINDBK.BSE',
]
INDIA_CONSUMER = [
    'HINDUNILVR.BSE', 'ITC.BSE', 'NESTLEIND.BSE', 'BRITANNIA.BSE',
    'TATACONSUM.BSE', 'ASIANPAINT.BSE', 'TITAN.BSE',
]
INDIA_INDUSTRIAL = [
    'RELIANCE.BSE', 'LT.BSE', 'MARUTI.BSE', 'TATAMOTORS.BSE',
    'EICHERMOT.BSE', 'ULTRACEMCO.BSE', 'GRASIM.BSE',
]
INDIA_HEALTH = ['SUNPHARMA.BSE', 'DRREDDY.BSE', 'CIPLA.BSE', 'DIVISLAB.BSE', 'APOLLOHOSP.BSE']
INDIA_ENERGY_METALS = [
    'ONGC.BSE', 'NTPC.BSE', 'POWERGRID.BSE', 'BPCL.BSE', 'COALINDIA.BSE',
    'TATASTEEL.BSE', 'HINDALCO.BSE', 'JSWSTEEL.BSE',
]
INDIA_TELECOM = ['BHARTIARTL.BSE', 'ADANIENT.BSE', 'ADANIPORTS.BSE']

INDIA_ALL = (
    INDIA_TECH + INDIA_FINANCE + INDIA_CONSUMER + INDIA_INDUSTRIAL
    + INDIA_HEALTH + INDIA_ENERGY_METALS + INDIA_TELECOM
)

# ── Forex Pairs ────────────────────────────────────────────
FOREX = [
    'EURUSD=X',    # Euro / US Dollar
    'GBPUSD=X',    # British Pound / US Dollar
    'USDJPY=X',    # US Dollar / Japanese Yen
    'USDINR=X',    # US Dollar / Indian Rupee
    'AUDUSD=X',    # Australian Dollar / US Dollar
    'USDCAD=X',    # US Dollar / Canadian Dollar
    'USDCHF=X',    # US Dollar / Swiss Franc
    'NZDUSD=X',    # New Zealand Dollar / US Dollar
    'EURGBP=X',    # Euro / British Pound
    'EURJPY=X',    # Euro / Japanese Yen
]

# ── Commodity Futures ──────────────────────────────────────
COMMODITIES = [
    'GC=F',    # Gold futures
    'SI=F',    # Silver futures
    'CL=F',    # WTI Crude Oil futures
    'NG=F',    # Natural Gas futures
    'HG=F',    # Copper futures
    'PL=F',    # Platinum futures
    'ZC=F',    # Corn futures
    'ZW=F',    # Wheat futures
    'ZS=F',    # Soybean futures
]

# ── Money Market / Treasury Yields ──────────────────────────
MONEY_MARKET = [
    '^TNX',    # 10-Year Treasury Yield
    '^IRX',    # 13-Week Treasury Bill Yield
    '^FVX',    # 5-Year Treasury Yield
    '^TYX',    # 30-Year Treasury Yield
]

# ── Cryptocurrency ──────────────────────────────────────────
CRYPTO = [
    'BTC-USD',     # Bitcoin
    'ETH-USD',     # Ethereum
    'SOL-USD',     # Solana
    'BNB-USD',     # Binance Coin
    'XRP-USD',     # Ripple
    'ADA-USD',     # Cardano
    'DOGE-USD',    # Dogecoin
    'AVAX-USD',    # Avalanche
    'DOT-USD',     # Polkadot
    'LINK-USD',    # Chainlink
]

ALL_SYMBOLS = (
    TECH + FINANCE + CONSUMER + HEALTH + INDUSTRIAL + ENERGY
    + INDEX_ETFS + SECTOR_ETFS + BOND_ETFS + DIVIDEND_ETFS
    + HEDGE_FUND_ALTS + REAL_ESTATE_COMMODITY + INTERNATIONAL
    + INDIA_ALL
    + FOREX + COMMODITIES + MONEY_MARKET + CRYPTO
)

QUICK_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'SPY', 'NVDA', 'JPM', 'AMZN', 'TSLA', 'META', 'QQQ',
    'EURUSD=X', 'GC=F', '^TNX', 'BTC-USD',
]


def parse_args():
    parser = argparse.ArgumentParser(description='AI Stock GPT Model Training')
    parser.add_argument('--tune', action='store_true', help='Run Optuna hyperparameter tuning before training')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark comparison report')
    parser.add_argument('--symbols', type=str, default=None, help='Comma-separated symbols to train')
    parser.add_argument('--quick', action='store_true', help='Train only 10 symbols for quick testing')
    parser.add_argument('--discover', action='store_true',
                        help='Auto-discover new symbols from Alpha Vantage LISTING_STATUS and add to training')
    parser.add_argument('--tune-trials', type=int, default=30, help='Number of Optuna trials per model (default: 30)')
    parser.add_argument('--tune-timeout', type=int, default=300, help='Optuna timeout in seconds (default: 300)')
    return parser.parse_args()


def discover_new_symbols(av_collector):
    """Use Alpha Vantage LISTING_STATUS to find new US/India symbols not in ALL_SYMBOLS."""
    all_known = set(s.upper() for s in ALL_SYMBOLS)

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
        logger.info(f"  New US (first 30): {new_us[:30]}")
    if new_india:
        logger.info(f"  New India: {new_india}")

    return new_us + new_india


def train_symbols(symbols, pipeline, predictor, tune=False, tune_trials=30, tune_timeout=300):
    """Train models for a list of symbols. Optionally run Optuna tuning first."""
    tuner = None
    if tune:
        try:
            from backend.ml.hyperparam_tuner import HyperparamTuner
            tuner = HyperparamTuner(params_dir='models')
            logger.info("Optuna hyperparameter tuning ENABLED")
        except ImportError:
            logger.warning("Optuna not available, skipping hyperparameter tuning")

    results = {}
    total_start = time.time()

    for i, symbol in enumerate(symbols, 1):
        logger.info(f"\n[{i}/{len(symbols)}] Training {symbol}...")
        start = time.time()

        try:
            # Build features
            logger.info(f"  Building features for {symbol}...")
            features = pipeline.build_features(symbol, lookback_days=756)
            logger.info(f"  Features shape: {features.shape}")

            # Optuna tuning (optional)
            tuned_params = None
            if tuner:
                # Check for cached params first
                cached = tuner.load_params(symbol)
                if cached:
                    logger.info(f"  Using cached tuned params for {symbol}")
                    tuned_params = cached
                else:
                    logger.info(f"  Running Optuna tuning for {symbol}...")
                    target = predictor._create_target(features)
                    tuned_params = tuner.tune(
                        symbol, features, target.values,
                        n_trials=tune_trials, timeout=tune_timeout,
                    )

            # Train model
            logger.info(f"  Training ensemble model...")
            metrics = predictor.train(symbol, features, tuned_params=tuned_params)

            elapsed = time.time() - start
            results[symbol] = {
                'status': 'success',
                'accuracy': metrics['accuracy'],
                'calibrated_accuracy': metrics.get('calibrated_accuracy', metrics['accuracy']),
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'n_features': metrics.get('n_features', 0),
                'ensemble': metrics.get('ensemble', False),
                'elapsed': elapsed,
            }

            logger.info(f"  {symbol} done in {elapsed:.1f}s")
            logger.info(f"  Accuracy: {metrics['accuracy']:.2%}  Calibrated: {metrics.get('calibrated_accuracy', 0):.2%}")
            logger.info(f"  Precision: {metrics['precision']:.2%}  Recall: {metrics['recall']:.2%}  F1: {metrics['f1']:.2%}")
            logger.info(f"  Features: {metrics.get('n_features', '?')}  Ensemble: {metrics.get('ensemble', False)}")
            logger.info(f"  Top features: {list(metrics.get('feature_importance', {}).keys())[:5]}")

            # Run prediction on latest data
            prediction = predictor.predict(symbol, features)
            logger.info(f"  Latest signal: {prediction['direction']} "
                        f"(prob={prediction['probability']:.2%}, "
                        f"confidence={prediction['confidence']:.2%})")

        except Exception as e:
            elapsed = time.time() - start
            results[symbol] = {'status': 'failed', 'error': str(e), 'elapsed': elapsed}
            logger.error(f"  FAILED: {e}")

    total_elapsed = time.time() - total_start
    return results, total_elapsed


def print_summary(results, total_elapsed):
    """Print training summary."""
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 70)

    succeeded = [s for s, r in results.items() if r['status'] == 'success']
    failed = [s for s, r in results.items() if r['status'] == 'failed']

    logger.info(f"Total: {len(results)} | Succeeded: {len(succeeded)} | Failed: {len(failed)}")
    logger.info(f"Total time: {total_elapsed:.1f}s")

    if succeeded:
        accs = [results[s]['accuracy'] for s in succeeded]
        cal_accs = [results[s].get('calibrated_accuracy', results[s]['accuracy']) for s in succeeded]
        f1s = [results[s]['f1'] for s in succeeded]
        precs = [results[s]['precision'] for s in succeeded]
        recs = [results[s]['recall'] for s in succeeded]

        import numpy as np
        logger.info(f"\nAccuracy     - mean: {np.mean(accs):.2%}  median: {np.median(accs):.2%}  std: {np.std(accs):.2%}")
        logger.info(f"Cal. Accuracy- mean: {np.mean(cal_accs):.2%}  median: {np.median(cal_accs):.2%}")
        logger.info(f"Precision    - mean: {np.mean(precs):.2%}  median: {np.median(precs):.2%}")
        logger.info(f"Recall       - mean: {np.mean(recs):.2%}  median: {np.median(recs):.2%}")
        logger.info(f"F1           - mean: {np.mean(f1s):.2%}  median: {np.median(f1s):.2%}")

        # Accuracy distribution
        bins = {'<50%': 0, '50-60%': 0, '60-70%': 0, '70-80%': 0, '80-90%': 0, '>90%': 0}
        for a in accs:
            if a < 0.5: bins['<50%'] += 1
            elif a < 0.6: bins['50-60%'] += 1
            elif a < 0.7: bins['60-70%'] += 1
            elif a < 0.8: bins['70-80%'] += 1
            elif a < 0.9: bins['80-90%'] += 1
            else: bins['>90%'] += 1
        logger.info(f"\nAccuracy distribution: {bins}")

    logger.info(f"\nPer-symbol results (sorted by accuracy):")
    sorted_results = sorted(
        [(s, r) for s, r in results.items() if r['status'] == 'success'],
        key=lambda x: x[1]['accuracy'], reverse=True
    )
    for symbol, r in sorted_results:
        ens = "E" if r.get('ensemble') else " "
        logger.info(
            f"  {symbol:20s}  acc={r['accuracy']:.2%}  cal={r.get('calibrated_accuracy', 0):.2%}  "
            f"f1={r['f1']:.2%}  prec={r['precision']:.2%}  rec={r['recall']:.2%}  "
            f"feat={r.get('n_features', '?')}  [{ens}]  ({r['elapsed']:.1f}s)"
        )

    if failed:
        logger.info(f"\nFailed symbols:")
        for symbol in failed:
            logger.info(f"  {symbol:20s}  {results[symbol].get('error', 'unknown')}")

    logger.info(f"\nModels saved to: {os.path.abspath('models')}/")

    # Save results to JSON for benchmarking
    report_path = os.path.join('models', 'training_report.json')
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_symbols': len(results),
        'succeeded': len(succeeded),
        'failed': len(failed),
        'total_time': total_elapsed,
        'results': results,
    }
    if succeeded:
        report['avg_accuracy'] = float(np.mean(accs))
        report['avg_calibrated_accuracy'] = float(np.mean(cal_accs))
        report['avg_f1'] = float(np.mean(f1s))
        report['median_accuracy'] = float(np.median(accs))

    try:
        import numpy as np
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Report saved to: {os.path.abspath(report_path)}")
    except Exception as e:
        logger.warning(f"Failed to save report: {e}")


def run_benchmark():
    """Compare current training report with previous run."""
    report_path = os.path.join('models', 'training_report.json')
    prev_path = os.path.join('models', 'training_report_prev.json')

    if not os.path.exists(report_path):
        logger.error("No training report found. Run training first.")
        return

    with open(report_path, 'r') as f:
        current = json.load(f)

    logger.info("=" * 70)
    logger.info("BENCHMARK REPORT")
    logger.info("=" * 70)
    logger.info(f"Current run: {current.get('timestamp', 'unknown')}")
    logger.info(f"  Symbols: {current['succeeded']}/{current['total_symbols']}")
    logger.info(f"  Avg Accuracy: {current.get('avg_accuracy', 0):.2%}")
    logger.info(f"  Avg Cal. Accuracy: {current.get('avg_calibrated_accuracy', 0):.2%}")
    logger.info(f"  Avg F1: {current.get('avg_f1', 0):.2%}")
    logger.info(f"  Median Accuracy: {current.get('median_accuracy', 0):.2%}")

    if os.path.exists(prev_path):
        with open(prev_path, 'r') as f:
            prev = json.load(f)

        logger.info(f"\nPrevious run: {prev.get('timestamp', 'unknown')}")
        logger.info(f"  Avg Accuracy: {prev.get('avg_accuracy', 0):.2%}")
        logger.info(f"  Avg F1: {prev.get('avg_f1', 0):.2%}")

        delta_acc = current.get('avg_accuracy', 0) - prev.get('avg_accuracy', 0)
        delta_f1 = current.get('avg_f1', 0) - prev.get('avg_f1', 0)
        logger.info(f"\n  Accuracy change: {delta_acc:+.2%}")
        logger.info(f"  F1 change:       {delta_f1:+.2%}")

        # Per-symbol comparison
        curr_results = current.get('results', {})
        prev_results = prev.get('results', {})
        common = set(curr_results.keys()) & set(prev_results.keys())

        improved = []
        degraded = []
        for s in common:
            cr = curr_results[s]
            pr = prev_results[s]
            if cr.get('status') == 'success' and pr.get('status') == 'success':
                delta = cr['accuracy'] - pr['accuracy']
                if delta > 0.01:
                    improved.append((s, delta))
                elif delta < -0.01:
                    degraded.append((s, delta))

        if improved:
            improved.sort(key=lambda x: x[1], reverse=True)
            logger.info(f"\nImproved ({len(improved)} symbols):")
            for s, d in improved[:10]:
                logger.info(f"  {s:20s}  {d:+.2%}")

        if degraded:
            degraded.sort(key=lambda x: x[1])
            logger.info(f"\nDegraded ({len(degraded)} symbols):")
            for s, d in degraded[:10]:
                logger.info(f"  {s:20s}  {d:+.2%}")
    else:
        logger.info("\nNo previous report found for comparison.")
        logger.info("Tip: Copy training_report.json to training_report_prev.json before retraining.")


def main():
    args = parse_args()

    if args.benchmark:
        run_benchmark()
        return

    logger.info("=" * 70)
    logger.info("AI Stock GPT - Enhanced Model Training v2")
    logger.info("=" * 70)

    os.makedirs('models', exist_ok=True)

    # Backup previous report for benchmarking
    report_path = os.path.join('models', 'training_report.json')
    prev_path = os.path.join('models', 'training_report_prev.json')
    if os.path.exists(report_path):
        import shutil
        shutil.copy2(report_path, prev_path)
        logger.info("Previous training report backed up for benchmarking")

    av_collector = AlphaVantageCollector()

    # Select symbols
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
    elif args.quick:
        symbols = QUICK_SYMBOLS
    elif args.discover:
        new_syms = discover_new_symbols(av_collector)
        symbols = list(ALL_SYMBOLS) + new_syms
        # Deduplicate
        seen = set()
        deduped = []
        for s in symbols:
            if s.upper() not in seen:
                seen.add(s.upper())
                deduped.append(s)
        symbols = deduped
        logger.info(f"Discovery mode: {len(symbols)} total symbols "
                    f"({len(ALL_SYMBOLS)} existing + {len(new_syms)} discovered)")
    else:
        symbols = ALL_SYMBOLS
    logger.info(f"Alpha Vantage API key: {'*' * max(0, len(av_collector.api_key) - 4)}{av_collector.api_key[-4:] if av_collector.api_key else 'NOT SET'}")
    logger.info(f"Total symbols to train: {len(symbols)}")
    logger.info(f"Optuna tuning: {'ENABLED' if args.tune else 'DISABLED'}")

    pipeline = FeaturePipeline(av_collector=av_collector)
    predictor = XGBoostPredictor(models_dir='models')

    results, total_elapsed = train_symbols(
        symbols, pipeline, predictor,
        tune=args.tune,
        tune_trials=args.tune_trials,
        tune_timeout=args.tune_timeout,
    )

    print_summary(results, total_elapsed)

    # Auto-benchmark if previous report exists
    if os.path.exists(prev_path):
        logger.info("")
        run_benchmark()


if __name__ == '__main__':
    main()
