#!/usr/bin/env python3
"""
Local training script for AI Stock GPT ML models.
Trains XGBoost models for a set of stock symbols using the feature pipeline.
"""

import os
import sys
import time
import logging

# Ensure backend is importable
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
# India - IT / Tech
INDIA_TECH = [
    'INFY.BSE', 'TCS.BSE', 'WIPRO.BSE', 'HCLTECH.BSE', 'TECHM.BSE',
]
# India - Banking / Finance
INDIA_FINANCE = [
    'HDFCBANK.BSE', 'ICICIBANK.BSE', 'SBIN.BSE', 'KOTAKBANK.BSE',
    'AXISBANK.BSE', 'BAJFINANCE.BSE', 'BAJAJFINSV.BSE', 'INDUSINDBK.BSE',
]
# India - Consumer / FMCG
INDIA_CONSUMER = [
    'HINDUNILVR.BSE', 'ITC.BSE', 'NESTLEIND.BSE', 'BRITANNIA.BSE',
    'TATACONSUM.BSE', 'ASIANPAINT.BSE', 'TITAN.BSE',
]
# India - Industrials / Auto
INDIA_INDUSTRIAL = [
    'RELIANCE.BSE', 'LT.BSE', 'MARUTI.BSE', 'TATAMOTORS.BSE',
    'EICHERMOT.BSE', 'ULTRACEMCO.BSE', 'GRASIM.BSE',
]
# India - Pharma / Healthcare
INDIA_HEALTH = [
    'SUNPHARMA.BSE', 'DRREDDY.BSE', 'CIPLA.BSE', 'DIVISLAB.BSE',
    'APOLLOHOSP.BSE',
]
# India - Energy / Metals / Mining
INDIA_ENERGY_METALS = [
    'ONGC.BSE', 'NTPC.BSE', 'POWERGRID.BSE', 'BPCL.BSE', 'COALINDIA.BSE',
    'TATASTEEL.BSE', 'HINDALCO.BSE', 'JSWSTEEL.BSE',
]
# India - Telecom / Conglomerate
INDIA_TELECOM = [
    'BHARTIARTL.BSE', 'ADANIENT.BSE', 'ADANIPORTS.BSE',
]

INDIA_ALL = (
    INDIA_TECH + INDIA_FINANCE + INDIA_CONSUMER + INDIA_INDUSTRIAL
    + INDIA_HEALTH + INDIA_ENERGY_METALS + INDIA_TELECOM
)

SYMBOLS = (
    TECH + FINANCE + CONSUMER + HEALTH + INDUSTRIAL + ENERGY
    + INDEX_ETFS + SECTOR_ETFS + BOND_ETFS + DIVIDEND_ETFS
    + HEDGE_FUND_ALTS + REAL_ESTATE_COMMODITY + INTERNATIONAL
    + INDIA_ALL
)


def main():
    logger.info("=" * 60)
    logger.info("AI Stock GPT - Local Model Training")
    logger.info("=" * 60)

    os.makedirs('models', exist_ok=True)

    av_collector = AlphaVantageCollector()
    logger.info(f"Alpha Vantage API key: {'*' * max(0, len(av_collector.api_key) - 4)}{av_collector.api_key[-4:] if av_collector.api_key else 'NOT SET'}")
    logger.info(f"Total symbols to train: {len(SYMBOLS)} ({len(SYMBOLS) - len(INDIA_ALL)} US + {len(INDIA_ALL)} India)")

    pipeline = FeaturePipeline(av_collector=av_collector)
    predictor = XGBoostPredictor(models_dir='models')

    results = {}
    total_start = time.time()

    for i, symbol in enumerate(SYMBOLS, 1):
        logger.info(f"\n[{i}/{len(SYMBOLS)}] Training {symbol}...")
        start = time.time()

        try:
            # Build features
            logger.info(f"  Building features for {symbol}...")
            features = pipeline.build_features(symbol, lookback_days=756)
            logger.info(f"  Features shape: {features.shape}")

            # Train model
            logger.info(f"  Training XGBoost model...")
            metrics = predictor.train(symbol, features)

            elapsed = time.time() - start
            results[symbol] = {
                'status': 'success',
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'elapsed': elapsed,
            }

            logger.info(f"  {symbol} done in {elapsed:.1f}s")
            logger.info(f"  Accuracy: {metrics['accuracy']:.2%}")
            logger.info(f"  Precision: {metrics['precision']:.2%}")
            logger.info(f"  Recall: {metrics['recall']:.2%}")
            logger.info(f"  F1: {metrics['f1']:.2%}")
            logger.info(f"  Top features: {list(metrics['feature_importance'].keys())[:5]}")

            # Run prediction on latest data
            prediction = predictor.predict(symbol, features)
            logger.info(f"  Latest signal: {prediction['direction']} "
                        f"(prob={prediction['probability']:.2%}, "
                        f"confidence={prediction['confidence']:.2%})")

        except Exception as e:
            elapsed = time.time() - start
            results[symbol] = {'status': 'failed', 'error': str(e), 'elapsed': elapsed}
            logger.error(f"  FAILED: {e}")

    # Summary
    total_elapsed = time.time() - total_start
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)

    succeeded = sum(1 for r in results.values() if r['status'] == 'success')
    failed = sum(1 for r in results.values() if r['status'] == 'failed')

    logger.info(f"Total: {len(SYMBOLS)} | Succeeded: {succeeded} | Failed: {failed}")
    logger.info(f"Total time: {total_elapsed:.1f}s")

    if succeeded > 0:
        avg_acc = sum(r['accuracy'] for r in results.values() if r['status'] == 'success') / succeeded
        avg_f1 = sum(r['f1'] for r in results.values() if r['status'] == 'success') / succeeded
        logger.info(f"Avg accuracy: {avg_acc:.2%}")
        logger.info(f"Avg F1: {avg_f1:.2%}")

    logger.info("\nPer-symbol results:")
    for symbol, r in results.items():
        if r['status'] == 'success':
            logger.info(f"  {symbol:6s}  acc={r['accuracy']:.2%}  f1={r['f1']:.2%}  ({r['elapsed']:.1f}s)")
        else:
            logger.info(f"  {symbol:6s}  FAILED: {r['error']}")

    logger.info(f"\nModels saved to: {os.path.abspath('models')}/")


if __name__ == '__main__':
    main()
