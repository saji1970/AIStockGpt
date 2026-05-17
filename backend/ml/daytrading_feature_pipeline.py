"""
Day-trading feature engineering pipeline for AI Stock GPT.

Computes intraday features from 5-minute OHLCV bars sourced via
Alpha Vantage TIME_SERIES_INTRADAY.  Features are designed for
short-horizon (next 6-12 bars / 30-60 min) directional prediction.

Feature groups:
  - VWAP and price-vs-VWAP
  - Intraday momentum (multi-bar returns)
  - Fast RSI / MACD / Bollinger / Stochastic / ADX / ATR
  - Volume profile (relative volume, volume spikes)
  - Time-of-day cyclical encoding
  - Previous-session reference levels (high/low/close)
  - Overnight gap
  - Candlestick shape features
  - Crossover signals
  - Mean-reversion z-scores
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DayTradingFeaturePipeline:
    """Build features from intraday 5-min bars for day-trading models."""

    def __init__(self, av_collector=None):
        self.av_collector = av_collector

    # ── public API ──────────────────────────────────────────────

    def build_features(
        self,
        symbol: str,
        months: int = 3,
        interval: str = "5min",
    ) -> pd.DataFrame:
        """Fetch intraday data and compute all day-trading features.

        Args:
            symbol: Ticker symbol.
            months: How many months of intraday history to use.
            interval: Bar interval (default '5min').

        Returns:
            DataFrame indexed by datetime with all features computed.
        """
        df = self._fetch_intraday(symbol, months, interval)

        if df is None or df.empty:
            raise ValueError(f"No intraday data available for {symbol}")

        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        for col in ["open", "high", "low", "close"]:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        if "volume" not in df.columns:
            df["volume"] = 0

        has_volume = df["volume"].sum() > 0

        # Compute feature groups
        df = self._add_vwap(df, has_volume)
        df = self._add_intraday_momentum(df)
        df = self._add_rsi(df)
        df = self._add_macd(df)
        df = self._add_bollinger(df)
        df = self._add_stochastic(df)
        df = self._add_atr(df)
        df = self._add_adx(df)
        if has_volume:
            df = self._add_volume_features(df)
        df = self._add_candlestick_features(df)
        df = self._add_crossover_signals(df)
        df = self._add_mean_reversion(df)
        df = self._add_session_reference_levels(df)
        df = self._add_overnight_gap(df)
        df = self._add_time_features(df)
        df = self._add_lag_features(df, has_volume)

        # Clean up
        all_nan = [c for c in df.columns if df[c].isna().all()]
        if all_nan:
            df = df.drop(columns=all_nan)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()

        logger.info(f"Day-trading features for {symbol}: {df.shape}")
        return df

    # ── data fetching ───────────────────────────────────────────

    def _fetch_intraday(
        self, symbol: str, months: int, interval: str
    ) -> Optional[pd.DataFrame]:
        if self.av_collector is None:
            raise ValueError("AlphaVantageCollector is required for intraday data")
        return self.av_collector.get_intraday_history(
            symbol, months=months, interval=interval
        )

    # ── VWAP ────────────────────────────────────────────────────

    @staticmethod
    def _add_vwap(df: pd.DataFrame, has_volume: bool) -> pd.DataFrame:
        """Cumulative intraday VWAP, resetting each trading day."""
        if not has_volume:
            df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3
            df["price_vs_vwap"] = 0.0
            return df

        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        tp_vol = typical_price * df["volume"]

        # Group by trading date
        dates = df.index.date
        cum_tp_vol = tp_vol.groupby(dates).cumsum()
        cum_vol = df["volume"].groupby(dates).cumsum().replace(0, np.nan)

        df["vwap"] = cum_tp_vol / cum_vol
        df["price_vs_vwap"] = (df["close"] - df["vwap"]) / df["vwap"].replace(0, np.nan)

        # Distance from VWAP in ATR units (if ATR exists later, this is raw for now)
        df["vwap_dist_pct"] = (df["close"] - df["vwap"]) / df["close"].replace(0, np.nan)

        return df

    # ── intraday momentum ──────────────────────────────────────

    @staticmethod
    def _add_intraday_momentum(df: pd.DataFrame) -> pd.DataFrame:
        """Multi-bar returns for intraday momentum signals."""
        close = df["close"]

        # Returns over various bar counts
        for bars in [1, 3, 6, 12, 24, 48]:
            df[f"return_{bars}bar"] = close.pct_change(bars)

        # Momentum acceleration
        df["momentum_accel_6"] = df["return_6bar"] - df["return_6bar"].shift(6)
        df["momentum_accel_12"] = df["return_12bar"] - df["return_12bar"].shift(12)

        # Rate of change
        df["roc_12bar"] = close.pct_change(12)
        df["roc_24bar"] = close.pct_change(24)

        return df

    # ── RSI ─────────────────────────────────────────────────────

    @staticmethod
    def _add_rsi(df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        delta = close.diff()

        # Fast RSI for intraday
        for period in [7, 14]:
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss.replace(0, np.nan)
            df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        return df

    # ── MACD ────────────────────────────────────────────────────

    @staticmethod
    def _add_macd(df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()

        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Faster MACD for scalping (5/13/4)
        fast_ema_5 = close.ewm(span=5, adjust=False).mean()
        fast_ema_13 = close.ewm(span=13, adjust=False).mean()
        df["macd_fast"] = fast_ema_5 - fast_ema_13
        df["macd_fast_signal"] = df["macd_fast"].ewm(span=4, adjust=False).mean()
        df["macd_fast_hist"] = df["macd_fast"] - df["macd_fast_signal"]

        return df

    # ── Bollinger Bands ─────────────────────────────────────────

    @staticmethod
    def _add_bollinger(df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]

        df["bb_middle"] = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
        df["bb_lower"] = df["bb_middle"] - (bb_std * 2)

        bb_width = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
        df["bb_width_pct"] = bb_width / df["bb_middle"].replace(0, np.nan)
        df["bb_position"] = (close - df["bb_lower"]) / bb_width

        return df

    # ── Stochastic ──────────────────────────────────────────────

    @staticmethod
    def _add_stochastic(df: pd.DataFrame) -> pd.DataFrame:
        high, low, close = df["high"], df["low"], df["close"]

        for period in [14, 7]:
            low_n = low.rolling(period).min()
            high_n = high.rolling(period).max()
            k = 100 * (close - low_n) / (high_n - low_n).replace(0, np.nan)
            d = k.rolling(3).mean()
            df[f"stoch_k_{period}"] = k
            df[f"stoch_d_{period}"] = d

        return df

    # ── ATR ─────────────────────────────────────────────────────

    @staticmethod
    def _add_atr(df: pd.DataFrame) -> pd.DataFrame:
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        df["atr_14"] = tr.rolling(14).mean()
        df["atr_7"] = tr.rolling(7).mean()
        df["atr_ratio"] = df["atr_14"] / close.replace(0, np.nan)

        # ATR expansion/contraction
        df["atr_expansion"] = df["atr_7"] / df["atr_14"].replace(0, np.nan)

        return df

    # ── ADX ─────────────────────────────────────────────────────

    @staticmethod
    def _add_adx(df: pd.DataFrame) -> pd.DataFrame:
        high, low, close = df["high"], df["low"], df["close"]

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr_14 = tr.rolling(14).mean().replace(0, np.nan)
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)

        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        df["adx_14"] = dx.rolling(14).mean()

        return df

    # ── volume features ─────────────────────────────────────────

    @staticmethod
    def _add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
        volume = df["volume"]

        # Relative volume vs 20-bar average
        vol_ma20 = volume.rolling(20).mean().replace(0, np.nan)
        df["relative_volume"] = volume / vol_ma20

        # Volume spike (> 2x average)
        df["volume_spike"] = (df["relative_volume"] > 2.0).astype(float)

        # Volume momentum
        df["volume_change_6bar"] = volume.pct_change(6)

        # OBV
        direction = np.sign(df["close"].diff())
        df["obv"] = (direction * volume).fillna(0).cumsum()

        # Volume-price divergence
        price_dir = np.sign(df["close"].pct_change(6))
        vol_dir = np.sign(volume.pct_change(6))
        df["vol_price_divergence"] = (price_dir != vol_dir).astype(float)

        # Volume-weighted close position
        vol_ma5 = volume.rolling(5).mean().replace(0, np.nan)
        df["vol_body_confirm"] = np.sign(df["close"] - df["open"]) * (volume / vol_ma5)

        return df

    # ── candlestick features ────────────────────────────────────

    @staticmethod
    def _add_candlestick_features(df: pd.DataFrame) -> pd.DataFrame:
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        day_range = (h - l).replace(0, np.nan)

        df["body_pct"] = (c - o) / o.replace(0, np.nan)
        df["body_range_ratio"] = (c - o).abs() / day_range

        body_top = pd.concat([c, o], axis=1).max(axis=1)
        body_bottom = pd.concat([c, o], axis=1).min(axis=1)
        df["upper_shadow_pct"] = (h - body_top) / day_range
        df["lower_shadow_pct"] = (body_bottom - l) / day_range

        df["close_position"] = (c - l) / day_range
        df["range_pct"] = (h - l) / c.replace(0, np.nan)

        # Range expansion vs 12-bar average
        avg_range_12 = (h - l).rolling(12).mean().replace(0, np.nan)
        df["range_expansion"] = (h - l) / avg_range_12

        return df

    # ── crossover signals ───────────────────────────────────────

    @staticmethod
    def _add_crossover_signals(df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]

        ema_5 = close.ewm(span=5, adjust=False).mean()
        ema_13 = close.ewm(span=13, adjust=False).mean()
        ema_21 = close.ewm(span=21, adjust=False).mean()

        df["ema_5_13_cross"] = np.where(ema_5 > ema_13, 1, -1).astype(float)
        df["ema_5_21_cross"] = np.where(ema_5 > ema_21, 1, -1).astype(float)

        # MACD crossover
        if "macd" in df.columns and "macd_signal" in df.columns:
            df["macd_cross"] = np.where(df["macd"] > df["macd_signal"], 1, -1).astype(float)

        # Fast MACD crossover
        if "macd_fast" in df.columns and "macd_fast_signal" in df.columns:
            df["macd_fast_cross"] = np.where(
                df["macd_fast"] > df["macd_fast_signal"], 1, -1
            ).astype(float)

        # Price vs VWAP
        if "vwap" in df.columns:
            df["price_above_vwap"] = (close > df["vwap"]).astype(float)

        # Stochastic crossover
        if "stoch_k_14" in df.columns and "stoch_d_14" in df.columns:
            df["stoch_cross"] = np.where(
                df["stoch_k_14"] > df["stoch_d_14"], 1, -1
            ).astype(float)

        return df

    # ── mean-reversion ──────────────────────────────────────────

    @staticmethod
    def _add_mean_reversion(df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]

        # Z-score from 20-bar rolling mean
        rm20 = close.rolling(20).mean()
        rs20 = close.rolling(20).std().replace(0, np.nan)
        df["zscore_20"] = (close - rm20) / rs20

        rm50 = close.rolling(50).mean()
        rs50 = close.rolling(50).std().replace(0, np.nan)
        df["zscore_50"] = (close - rm50) / rs50

        # Distance from 20-bar EMA
        ema_20 = close.ewm(span=20, adjust=False).mean()
        df["dist_ema_20"] = (close - ema_20) / ema_20.replace(0, np.nan)

        # RSI mean-reversion
        if "rsi_14" in df.columns:
            df["rsi_zscore"] = (
                (df["rsi_14"] - df["rsi_14"].rolling(50).mean())
                / df["rsi_14"].rolling(50).std().replace(0, np.nan)
            )

        return df

    # ── previous session reference levels ───────────────────────

    @staticmethod
    def _add_session_reference_levels(df: pd.DataFrame) -> pd.DataFrame:
        """Previous day's high, low, close as support/resistance levels."""
        dates = df.index.date
        daily_high = df["high"].groupby(dates).transform("max")
        daily_low = df["low"].groupby(dates).transform("min")
        daily_close = df["close"].groupby(dates).transform("last")

        # Shift by one trading day
        prev_high = daily_high.groupby(dates).first().shift(1)
        prev_low = daily_low.groupby(dates).first().shift(1)
        prev_close = daily_close.groupby(dates).first().shift(1)

        # Map back to intraday bars
        date_series = pd.Series(dates, index=df.index)
        df["prev_day_high"] = date_series.map(
            pd.Series(prev_high.values, index=prev_high.index)
        ).astype(float)
        df["prev_day_low"] = date_series.map(
            pd.Series(prev_low.values, index=prev_low.index)
        ).astype(float)
        df["prev_day_close"] = date_series.map(
            pd.Series(prev_close.values, index=prev_close.index)
        ).astype(float)

        # Price relative to previous levels
        close = df["close"]
        df["price_vs_prev_high"] = (close - df["prev_day_high"]) / df["prev_day_high"].replace(0, np.nan)
        df["price_vs_prev_low"] = (close - df["prev_day_low"]) / df["prev_day_low"].replace(0, np.nan)
        df["price_vs_prev_close"] = (close - df["prev_day_close"]) / df["prev_day_close"].replace(0, np.nan)

        return df

    # ── overnight gap ───────────────────────────────────────────

    @staticmethod
    def _add_overnight_gap(df: pd.DataFrame) -> pd.DataFrame:
        """Gap between previous close and current day's open."""
        dates = df.index.date
        daily_open = df["open"].groupby(dates).transform("first")
        daily_close = df["close"].groupby(dates).transform("last")

        prev_close = daily_close.groupby(dates).first().shift(1)
        day_open = daily_open.groupby(dates).first()

        date_series = pd.Series(dates, index=df.index)
        mapped_prev_close = date_series.map(
            pd.Series(prev_close.values, index=prev_close.index)
        ).astype(float)
        mapped_day_open = date_series.map(
            pd.Series(day_open.values, index=day_open.index)
        ).astype(float)

        df["overnight_gap_pct"] = (
            (mapped_day_open - mapped_prev_close) / mapped_prev_close.replace(0, np.nan)
        )

        # Gap direction
        df["gap_direction"] = np.sign(df["overnight_gap_pct"]).fillna(0)

        return df

    # ── time-of-day features ────────────────────────────────────

    @staticmethod
    def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """Cyclical time-of-day encoding and session flags."""
        minutes_since_open = df.index.hour * 60 + df.index.minute
        # Regular trading: 9:30 AM to 4:00 PM ET = 570 min to 960 min
        trading_minutes = 390  # 6.5 hours

        # Cyclical encoding of time within trading day
        normalized_time = (minutes_since_open - 570) / trading_minutes
        df["time_sin"] = np.sin(2 * np.pi * normalized_time)
        df["time_cos"] = np.cos(2 * np.pi * normalized_time)

        # Session flags (opening 30 min, midday, power hour)
        df["is_opening"] = ((minutes_since_open >= 570) & (minutes_since_open < 600)).astype(float)
        df["is_midday"] = ((minutes_since_open >= 720) & (minutes_since_open < 810)).astype(float)
        df["is_power_hour"] = ((minutes_since_open >= 900) & (minutes_since_open <= 960)).astype(float)

        # Day of week
        dow = df.index.dayofweek
        df["day_sin"] = np.sin(2 * np.pi * dow / 5)
        df["day_cos"] = np.cos(2 * np.pi * dow / 5)

        return df

    # ── lag features ────────────────────────────────────────────

    @staticmethod
    def _add_lag_features(df: pd.DataFrame, has_volume: bool) -> pd.DataFrame:
        """Lagged returns and volume features."""
        returns = df["close"].pct_change()

        df["return_lag_1"] = returns.shift(1)
        df["return_lag_2"] = returns.shift(2)
        df["return_lag_3"] = returns.shift(3)
        df["return_lag_6"] = returns.shift(6)
        df["return_lag_12"] = returns.shift(12)

        if has_volume:
            vol_returns = df["volume"].pct_change()
            df["vol_return_lag_1"] = vol_returns.shift(1)
            df["vol_return_lag_6"] = vol_returns.shift(6)

        # Return autocorrelation
        df["return_autocorr_6"] = returns.rolling(24).apply(
            lambda x: x.autocorr(lag=6) if len(x) > 6 else 0, raw=False
        )

        return df
