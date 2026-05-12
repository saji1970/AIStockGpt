"""
Alpha Vantage market data collector with lightweight TTL caching.
"""

import os
import time
import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class AlphaVantageCollector:
    """Fetches quote and daily OHLCV data from Alpha Vantage."""

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self) -> None:
        self.api_key = os.getenv("alphavantage_apikey") or os.getenv("ALPHA_VANTAGE_KEY", "")
        self.quote_ttl_seconds = 300
        self.history_ttl_seconds = 3600
        self._cache: Dict[Tuple[str, str], Tuple[float, Any]] = {}

    def _get_cached(self, namespace: str, symbol: str, ttl_seconds: int) -> Optional[Any]:
        key = (namespace, symbol.upper())
        entry = self._cache.get(key)
        if not entry:
            return None
        cached_at, data = entry
        if (time.time() - cached_at) <= ttl_seconds:
            return data
        self._cache.pop(key, None)
        return None

    def _set_cached(self, namespace: str, symbol: str, data: Any) -> None:
        self._cache[(namespace, symbol.upper())] = (time.time(), data)

    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return normalized quote data for a symbol."""
        if not self.api_key:
            return None

        cached = self._get_cached("quote", symbol, self.quote_ttl_seconds)
        if cached is not None:
            return cached

        try:
            response = requests.get(
                self.BASE_URL,
                params={
                    "function": "GLOBAL_QUOTE",
                    "symbol": symbol.upper(),
                    "apikey": self.api_key,
                },
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
            quote = payload.get("Global Quote", {})
            if not quote:
                logger.warning(f"Alpha Vantage quote unavailable for {symbol}: {payload}")
                return None

            price = float(quote.get("05. price", 0) or 0)
            previous_close = float(quote.get("08. previous close", 0) or 0)
            change = float(quote.get("09. change", 0) or 0)
            change_percent_raw = (quote.get("10. change percent", "0") or "0").replace("%", "")
            change_percent = float(change_percent_raw or 0)

            data = {
                "symbol": symbol.upper(),
                "price": round(price, 2),
                "previousClose": round(previous_close, 2),
                "change": round(change, 2),
                "changePercent": round(change_percent, 2),
                "high": round(float(quote.get("03. high", 0) or 0), 2),
                "low": round(float(quote.get("04. low", 0) or 0), 2),
                "volume": int(float(quote.get("06. volume", 0) or 0)),
                "name": symbol.upper(),
                "open": round(float(quote.get("02. open", 0) or 0), 2),
            }

            self._set_cached("quote", symbol, data)
            return data
        except Exception as exc:
            logger.warning(f"Alpha Vantage quote fetch failed for {symbol}: {exc}")
            return None

    def get_daily_history(self, symbol: str, days: int = 504) -> Optional[pd.DataFrame]:
        """Return daily OHLCV DataFrame indexed by date."""
        if not self.api_key:
            return None

        cached = self._get_cached("history", symbol, self.history_ttl_seconds)
        if cached is not None:
            return cached.tail(days).copy()

        try:
            series = {}
            for outputsize in ("full", "compact"):
                response = requests.get(
                    self.BASE_URL,
                    params={
                        "function": "TIME_SERIES_DAILY",
                        "symbol": symbol.upper(),
                        "outputsize": outputsize,
                        "apikey": self.api_key,
                    },
                    timeout=30,
                )
                response.raise_for_status()
                payload = response.json()
                series = payload.get("Time Series (Daily)", {})
                if series:
                    break
                if outputsize == "full":
                    logger.info(
                        f"Alpha Vantage full history unavailable for {symbol}; retrying compact output."
                    )
            if not series:
                logger.warning(f"Alpha Vantage daily history unavailable for {symbol}: {payload}")
                return None

            rows = []
            for date_str, values in series.items():
                rows.append(
                    {
                        "date": pd.to_datetime(date_str),
                        "open": float(values.get("1. open", 0) or 0),
                        "high": float(values.get("2. high", 0) or 0),
                        "low": float(values.get("3. low", 0) or 0),
                        "close": float(values.get("4. close", 0) or 0),
                        "volume": float(values.get("5. volume", 0) or 0),
                    }
                )

            df = pd.DataFrame(rows).sort_values("date").set_index("date")
            if df.empty:
                return None

            self._set_cached("history", symbol, df)
            return df.tail(days).copy()
        except Exception as exc:
            logger.warning(f"Alpha Vantage history fetch failed for {symbol}: {exc}")
            return None
