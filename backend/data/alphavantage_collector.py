"""
Alpha Vantage market data collector with lightweight TTL caching.
Supports daily OHLCV, real-time quotes, and intraday time-series
for day-trading pipelines.
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class AlphaVantageCollector:
    """Fetches quote, daily OHLCV, and intraday time-series data from Alpha Vantage."""

    BASE_URL = "https://www.alphavantage.co/query"
    INTRADAY_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "intraday")

    def __init__(self) -> None:
        self.api_key = os.getenv("alphavantage_apikey") or os.getenv("ALPHA_VANTAGE_KEY", "")
        self.quote_ttl_seconds = 300
        self.history_ttl_seconds = 3600
        self._cache: Dict[Tuple[str, str], Tuple[float, Any]] = {}
        os.makedirs(self.INTRADAY_CACHE_DIR, exist_ok=True)

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

    # ── intraday time-series ─────────────────────────────────────

    @staticmethod
    def _safe_filename(symbol: str) -> str:
        return symbol.replace("=", "_").replace("^", "_").replace("/", "_")

    def _intraday_cache_path(self, symbol: str, month: str, interval: str) -> str:
        return os.path.join(
            self.INTRADAY_CACHE_DIR,
            f"{self._safe_filename(symbol)}_{interval}_{month}.csv",
        )

    def get_intraday_month(
        self,
        symbol: str,
        month: str,
        interval: str = "5min",
        adjusted: bool = True,
    ) -> Optional[pd.DataFrame]:
        """Fetch one month of intraday OHLCV data.

        Args:
            symbol: Ticker (e.g. 'AAPL').
            month: YYYY-MM string (e.g. '2026-04').
            interval: '1min', '5min', '15min', '30min', or '60min'.
            adjusted: Split/dividend adjusted prices.

        Returns:
            DataFrame indexed by datetime with columns open/high/low/close/volume,
            or None on failure.
        """
        if not self.api_key:
            return None

        # Check disk cache first
        cache_path = self._intraday_cache_path(symbol, month, interval)
        if os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                if not df.empty:
                    logger.debug(f"Intraday cache hit: {symbol} {month} ({len(df)} bars)")
                    return df
            except Exception:
                pass

        try:
            response = requests.get(
                self.BASE_URL,
                params={
                    "function": "TIME_SERIES_INTRADAY",
                    "symbol": symbol.upper(),
                    "interval": interval,
                    "month": month,
                    "outputsize": "full",
                    "adjusted": str(adjusted).lower(),
                    "extended_hours": "false",
                    "apikey": self.api_key,
                },
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()

            # Check for API error messages
            if "Error Message" in payload or "Note" in payload:
                msg = payload.get("Error Message") or payload.get("Note", "")
                logger.warning(f"Alpha Vantage intraday error for {symbol} {month}: {msg}")
                return None

            series_key = f"Time Series ({interval})"
            series = payload.get(series_key, {})
            if not series:
                logger.warning(f"No intraday data for {symbol} {month}: keys={list(payload.keys())}")
                return None

            rows = []
            for ts_str, values in series.items():
                rows.append({
                    "datetime": pd.to_datetime(ts_str),
                    "open": float(values.get("1. open", 0) or 0),
                    "high": float(values.get("2. high", 0) or 0),
                    "low": float(values.get("3. low", 0) or 0),
                    "close": float(values.get("4. close", 0) or 0),
                    "volume": float(values.get("5. volume", 0) or 0),
                })

            df = pd.DataFrame(rows).sort_values("datetime").set_index("datetime")
            if df.empty:
                return None

            # Cache to disk
            df.to_csv(cache_path)
            logger.info(f"Fetched intraday {symbol} {month} {interval}: {len(df)} bars (cached)")
            return df

        except Exception as exc:
            logger.warning(f"Intraday fetch failed for {symbol} {month}: {exc}")
            return None

    def get_intraday_history(
        self,
        symbol: str,
        months: int = 3,
        interval: str = "5min",
        rate_limit_delay: float = 12.5,
    ) -> Optional[pd.DataFrame]:
        """Fetch multiple months of intraday data, concatenated.

        Uses disk caching so already-fetched months are not re-downloaded.
        Respects API rate limits with a configurable delay between requests.

        Args:
            symbol: Ticker.
            months: Number of months to fetch (going back from current).
            interval: Bar interval.
            rate_limit_delay: Seconds to wait between API calls (Alpha Vantage
                free tier allows ~5 calls/min).

        Returns:
            Concatenated DataFrame of intraday bars, or None.
        """
        if not self.api_key:
            return None

        # Build list of YYYY-MM strings
        now = datetime.now()
        month_list: List[str] = []
        for i in range(months):
            dt = now - timedelta(days=30 * i)
            month_str = dt.strftime("%Y-%m")
            if month_str not in month_list:
                month_list.append(month_str)

        frames: List[pd.DataFrame] = []
        api_calls_made = 0

        for month_str in sorted(month_list):
            # Check if already cached (no API call needed)
            cache_path = self._intraday_cache_path(symbol, month_str, interval)
            needs_api = not os.path.exists(cache_path)

            if needs_api and api_calls_made > 0:
                time.sleep(rate_limit_delay)

            df_month = self.get_intraday_month(symbol, month_str, interval)
            if df_month is not None and not df_month.empty:
                frames.append(df_month)

            if needs_api:
                api_calls_made += 1

        if not frames:
            logger.warning(f"No intraday data collected for {symbol}")
            return None

        combined = pd.concat(frames).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        logger.info(f"Intraday history for {symbol}: {len(combined)} bars over {len(frames)} months")
        return combined
