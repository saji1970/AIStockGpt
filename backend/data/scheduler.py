"""
APScheduler-based data refresh scheduler.
"""

import logging
from datetime import datetime

import yfinance as yf
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)


class DataScheduler:
    """APScheduler-based data refresh."""

    def __init__(self, db_manager, fred_collector=None, sentiment_analyzer=None):
        self.db_manager = db_manager
        self.fred_collector = fred_collector
        self.sentiment_analyzer = sentiment_analyzer
        self.scheduler = BackgroundScheduler()

    def start(self):
        """Start scheduler with periodic jobs."""
        # Daily at 18:00 ET (23:00 UTC): refresh FRED macro data and market data
        self.scheduler.add_job(
            self._daily_refresh,
            trigger=CronTrigger(hour=23, minute=0),
            id='daily_refresh',
            name='Daily FRED + market data refresh',
            replace_existing=True,
        )

        # Every 4 hours: refresh sentiment for top-held symbols
        self.scheduler.add_job(
            self._sentiment_refresh,
            trigger=IntervalTrigger(hours=4),
            id='sentiment_refresh',
            name='Sentiment refresh for top symbols',
            replace_existing=True,
        )

        self.scheduler.start()
        logger.info("Data scheduler started with 2 jobs")

    def stop(self):
        """Shutdown scheduler gracefully."""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            logger.info("Data scheduler stopped")

    def _daily_refresh(self):
        """Job: fetch FRED data, fetch market_data for tracked symbols via yfinance."""
        logger.info("Running daily data refresh...")

        # FRED data
        if self.fred_collector:
            try:
                count = self.fred_collector.save_to_db(self.db_manager)
                logger.info(f"FRED refresh: {count} records saved")
            except Exception as e:
                logger.error(f"FRED refresh failed: {e}")

        # Market data for tracked symbols
        try:
            symbols = self.db_manager.get_tracked_symbols()
            if not symbols:
                symbols = ['SPY', 'QQQ', 'DIA']  # Default indices

            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period='5d')
                    if hist.empty:
                        continue

                    for date, row in hist.iterrows():
                        ohlcv = {
                            'open': float(row.get('Open', 0)),
                            'high': float(row.get('High', 0)),
                            'low': float(row.get('Low', 0)),
                            'close': float(row.get('Close', 0)),
                            'volume': float(row.get('Volume', 0)),
                            'adjusted_close': float(row.get('Close', 0)),
                        }
                        self.db_manager.save_market_data(
                            symbol=symbol,
                            date=date.date() if hasattr(date, 'date') else date,
                            ohlcv=ohlcv,
                            source='yfinance',
                        )
                    logger.info(f"Market data refreshed for {symbol}")
                except Exception as e:
                    logger.warning(f"Failed to refresh market data for {symbol}: {e}")

        except Exception as e:
            logger.error(f"Market data refresh failed: {e}")

    def _sentiment_refresh(self):
        """Job: run SentimentAnalyzer on top-held symbols."""
        if not self.sentiment_analyzer:
            return

        try:
            symbols = self.db_manager.get_tracked_symbols()
            # Limit to top 10 most held symbols
            for symbol in symbols[:10]:
                try:
                    self.sentiment_analyzer.analyze_symbol(symbol)
                    logger.info(f"Sentiment refreshed for {symbol}")
                except Exception as e:
                    logger.warning(f"Sentiment refresh failed for {symbol}: {e}")
        except Exception as e:
            logger.error(f"Sentiment refresh failed: {e}")
