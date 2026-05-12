"""
Fetch macroeconomic indicators from FRED API.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class FREDCollector:
    """Fetch macroeconomic indicators from FRED API."""

    INDICATORS = {
        'DFF': 'Federal Funds Rate',
        'CPIAUCSL': 'CPI (inflation)',
        'UNRATE': 'Unemployment Rate',
        'GDP': 'Gross Domestic Product',
        'T10Y2Y': '10Y-2Y Treasury Spread (yield curve)',
    }

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('FRED_API_KEY', '')
        self._fred = None

    def _get_fred(self):
        """Lazy-load FRED client."""
        if self._fred is None:
            if not self.api_key:
                raise ValueError("FRED_API_KEY not set")
            from fredapi import Fred
            self._fred = Fred(api_key=self.api_key)
        return self._fred

    def fetch_all(self, start_date: str = None) -> Dict[str, pd.Series]:
        """Fetch all indicators. Returns {indicator_name: pd.Series}."""
        if start_date is None:
            start_date = (datetime.utcnow() - timedelta(days=365 * 2)).strftime('%Y-%m-%d')

        fred = self._get_fred()
        results = {}

        for indicator_id, description in self.INDICATORS.items():
            try:
                data = fred.get_series(indicator_id, observation_start=start_date)
                if data is not None and len(data) > 0:
                    results[indicator_id] = data
                    logger.info(f"Fetched {indicator_id} ({description}): {len(data)} observations")
            except Exception as e:
                logger.warning(f"Failed to fetch {indicator_id}: {e}")

        return results

    def save_to_db(self, db_manager) -> int:
        """Fetch and store in macro_indicators table. Returns count of new rows."""
        count = 0
        try:
            data = self.fetch_all()
            for indicator_name, series in data.items():
                for date, value in series.items():
                    if pd.notna(value):
                        result = db_manager.save_macro_indicator(
                            name=indicator_name,
                            date=date.date() if hasattr(date, 'date') else date,
                            value=float(value),
                            source='fred',
                        )
                        if result:
                            count += 1
            logger.info(f"Saved {count} macro indicator records to DB")
        except Exception as e:
            logger.error(f"Failed to save FRED data to DB: {e}")

        return count
