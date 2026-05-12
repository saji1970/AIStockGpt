"""
Fetch news and basic data from Finnhub API.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List

import requests

logger = logging.getLogger(__name__)


class FinnhubCollector:
    """Fetch news and basic data from Finnhub API."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('FINNHUB_API_KEY', '')
        self.base_url = 'https://finnhub.io/api/v1'

    def get_company_news(self, symbol: str, days: int = 7) -> List[Dict]:
        """Returns list of news items with headline, summary, source, datetime, url."""
        if not self.api_key:
            logger.warning("FINNHUB_API_KEY not set")
            return []

        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)

            params = {
                'symbol': symbol,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'token': self.api_key,
            }
            resp = requests.get(f'{self.base_url}/company-news', params=params, timeout=10)

            if resp.status_code != 200:
                logger.warning(f"Finnhub news API returned {resp.status_code}")
                return []

            data = resp.json()
            results = []
            for item in data[:20]:
                results.append({
                    'headline': item.get('headline', ''),
                    'summary': item.get('summary', ''),
                    'source': item.get('source', ''),
                    'datetime': datetime.fromtimestamp(item.get('datetime', 0)).isoformat(),
                    'url': item.get('url', ''),
                })

            return results
        except Exception as e:
            logger.error(f"Failed to fetch Finnhub news for {symbol}: {e}")
            return []

    def get_quote(self, symbol: str) -> Dict:
        """Returns current quote data."""
        if not self.api_key:
            return {}

        try:
            params = {'symbol': symbol, 'token': self.api_key}
            resp = requests.get(f'{self.base_url}/quote', params=params, timeout=10)

            if resp.status_code != 200:
                return {}

            data = resp.json()
            return {
                'current_price': data.get('c', 0),
                'change': data.get('d', 0),
                'percent_change': data.get('dp', 0),
                'high': data.get('h', 0),
                'low': data.get('l', 0),
                'open': data.get('o', 0),
                'previous_close': data.get('pc', 0),
            }
        except Exception as e:
            logger.error(f"Failed to fetch Finnhub quote for {symbol}: {e}")
            return {}
