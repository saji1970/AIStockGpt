"""
Feature engineering pipeline for AI Stock GPT.
Computes technical, volatility, momentum, macro indicators from price data.
"""

import logging
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Unified feature engineering: technical, volatility, momentum, macro."""

    def __init__(self, db_manager=None, av_collector=None):
        """Uses db_manager for macro data. Falls back to yfinance for price data."""
        self.db_manager = db_manager
        self.av_collector = av_collector

    def build_features(self, symbol: str, lookback_days: int = 504) -> pd.DataFrame:
        """
        Fetch price data via yfinance, compute all features, return clean DataFrame.

        Returns DataFrame indexed by date with ~50 feature columns.
        """
        # Fetch price data (Alpha Vantage first, yfinance fallback)
        df = pd.DataFrame()
        if self.av_collector:
            try:
                av_df = self.av_collector.get_daily_history(symbol, days=lookback_days)
                if av_df is not None and not av_df.empty:
                    df = av_df.copy()
                    logger.info(f"Feature data source for {symbol}: Alpha Vantage")
            except Exception as e:
                logger.warning(f"Alpha Vantage feature fetch failed for {symbol}: {e}")

        if df.empty:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{lookback_days}d")
            logger.info(f"Feature data source for {symbol}: yfinance")

        if df.empty:
            raise ValueError(f"No price data available for {symbol}")

        # Standardize column names
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]

        # Ensure we have OHLCV
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        # Technical indicators
        df = self._add_technical_indicators(df)

        # Volatility indicators
        df = self._add_volatility_indicators(df)

        # Momentum indicators
        df = self._add_momentum_indicators(df, symbol)

        # Lag features
        df = self._add_lag_features(df)

        # Rolling statistics
        df = self._add_rolling_stats(df)

        # Time features
        df = self._add_time_features(df)

        # Macro indicators
        df = self._add_macro_features(df)

        # Drop columns that are entirely NaN (e.g. macro data when DB unavailable)
        all_nan_cols = [c for c in df.columns if df[c].isna().all()]
        if all_nan_cols:
            df = df.drop(columns=all_nan_cols)
            logger.info(f"Dropped {len(all_nan_cols)} all-NaN columns: {all_nan_cols}")

        # Drop NaN rows
        df = df.dropna()

        logger.info(f"Built features for {symbol}: {df.shape}")
        return df

    def compute_indicators(self, symbol: str) -> Dict[str, Any]:
        """
        Compute latest indicator values for a symbol.
        Used by technical_analysis intent in chat.
        """
        try:
            df = self.build_features(symbol, lookback_days=120)

            if df.empty:
                return {}

            latest = df.iloc[-1]
            result = {
                'symbol': symbol,
                'date': str(df.index[-1].date()) if hasattr(df.index[-1], 'date') else str(df.index[-1]),
                'close': float(latest.get('close', 0)),
                'rsi_14': float(latest.get('rsi_14', 0)),
                'macd': float(latest.get('macd', 0)),
                'macd_signal': float(latest.get('macd_signal', 0)),
                'sma_20': float(latest.get('sma_20', 0)),
                'sma_50': float(latest.get('sma_50', 0)),
                'ema_12': float(latest.get('ema_12', 0)),
                'ema_26': float(latest.get('ema_26', 0)),
                'bb_upper': float(latest.get('bb_upper', 0)),
                'bb_lower': float(latest.get('bb_lower', 0)),
                'bb_middle': float(latest.get('bb_middle', 0)),
                'atr_14': float(latest.get('atr_14', 0)),
                'stoch_k': float(latest.get('stoch_k', 0)),
                'stoch_d': float(latest.get('stoch_d', 0)),
                'adx_14': float(latest.get('adx_14', 0)),
                'obv': float(latest.get('obv', 0)),
                'hist_vol_20': float(latest.get('hist_vol_20', 0)),
                'hist_vol_60': float(latest.get('hist_vol_60', 0)),
                'return_5d': float(latest.get('return_5d', 0)),
                'return_21d': float(latest.get('return_21d', 0)),
                'rolling_sharpe_21': float(latest.get('rolling_sharpe_21', 0)),
            }

            # Add signal interpretations
            rsi = result['rsi_14']
            if rsi > 70:
                result['rsi_signal'] = 'overbought'
            elif rsi < 30:
                result['rsi_signal'] = 'oversold'
            else:
                result['rsi_signal'] = 'neutral'

            if result['macd'] > result['macd_signal']:
                result['macd_signal_direction'] = 'bullish'
            else:
                result['macd_signal_direction'] = 'bearish'

            close = result['close']
            if close > result['bb_upper']:
                result['bb_signal'] = 'above upper band'
            elif close < result['bb_lower']:
                result['bb_signal'] = 'below lower band'
            else:
                result['bb_signal'] = 'within bands'

            return result
        except Exception as e:
            logger.error(f"Failed to compute indicators for {symbol}: {e}")
            return {}

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis indicators."""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # Simple Moving Averages
        df['sma_20'] = close.rolling(window=20).mean()
        df['sma_50'] = close.rolling(window=50).mean()

        # Exponential Moving Averages
        df['ema_12'] = close.ewm(span=12, adjust=False).mean()
        df['ema_26'] = close.ewm(span=26, adjust=False).mean()

        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['bb_middle'] = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

        # Average True Range (ATR)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(window=14).mean()

        # On-Balance Volume
        obv = [0]
        for i in range(1, len(df)):
            if close.iloc[i] > close.iloc[i - 1]:
                obv.append(obv[-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i - 1]:
                obv.append(obv[-1] - volume.iloc[i])
            else:
                obv.append(obv[-1])
        df['obv'] = obv

        # Stochastic Oscillator
        low_14 = low.rolling(window=14).min()
        high_14 = high.rolling(window=14).max()
        df['stoch_k'] = 100 * (close - low_14) / (high_14 - low_14).replace(0, np.nan)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

        # ADX (Average Directional Index)
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr_14 = tr.rolling(window=14).mean()
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr_14.replace(0, np.nan))
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr_14.replace(0, np.nan))
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        df['adx_14'] = dx.rolling(window=14).mean()

        return df

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        close = df['close']
        returns = close.pct_change()

        # Historical volatility (annualized)
        df['hist_vol_20'] = returns.rolling(window=20).std() * np.sqrt(252)
        df['hist_vol_60'] = returns.rolling(window=60).std() * np.sqrt(252)

        # Garman-Klass volatility
        log_hl = np.log(df['high'] / df['low']) ** 2
        log_co = np.log(df['close'] / df['open']) ** 2
        df['garman_klass'] = np.sqrt(
            (0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(window=20).mean() * 252
        )

        # ATR ratio (ATR / close)
        if 'atr_14' in df.columns:
            df['atr_ratio'] = df['atr_14'] / close

        # Volatility of volatility
        df['vol_of_vol'] = df['hist_vol_20'].rolling(window=20).std()

        return df

    def _add_momentum_indicators(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add momentum indicators."""
        close = df['close']
        returns = close.pct_change()

        # Multi-period returns
        df['return_5d'] = close.pct_change(5)
        df['return_21d'] = close.pct_change(21)
        df['return_63d'] = close.pct_change(63)
        df['return_252d'] = close.pct_change(252)

        # Beta vs SPY
        try:
            spy = yf.Ticker('SPY').history(period=f"{len(df) + 30}d")
            if not spy.empty:
                spy.columns = [c.lower().replace(' ', '_') for c in spy.columns]
                spy_returns = spy['close'].pct_change()
                # Align dates
                aligned = pd.DataFrame({
                    'stock': returns,
                    'spy': spy_returns
                }).dropna()
                if len(aligned) > 60:
                    rolling_cov = aligned['stock'].rolling(60).cov(aligned['spy'])
                    rolling_var = aligned['spy'].rolling(60).var()
                    beta = rolling_cov / rolling_var.replace(0, np.nan)
                    df['beta_spy'] = beta.reindex(df.index)
                else:
                    df['beta_spy'] = np.nan
            else:
                df['beta_spy'] = np.nan
        except Exception:
            df['beta_spy'] = np.nan

        # Rolling Sharpe ratio (21-day)
        df['rolling_sharpe_21'] = (
            returns.rolling(window=21).mean() / returns.rolling(window=21).std().replace(0, np.nan)
        ) * np.sqrt(252)

        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features."""
        df['close_lag_1'] = df['close'].shift(1)
        df['close_lag_5'] = df['close'].shift(5)
        df['volume_lag_1'] = df['volume'].shift(1)
        df['volume_lag_5'] = df['volume'].shift(5)
        return df

    def _add_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistics."""
        df['close_rolling_mean_20'] = df['close'].rolling(window=20).mean()
        df['close_rolling_std_20'] = df['close'].rolling(window=20).std()
        df['volume_rolling_mean_20'] = df['volume'].rolling(window=20).mean()
        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        idx = df.index
        df['day_of_week'] = idx.dayofweek
        df['month'] = idx.month
        df['quarter'] = idx.quarter
        return df

    def _add_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add macroeconomic features from the database."""
        macro_cols = {
            'DFF': 'fed_rate',
            'CPIAUCSL': 'cpi_yoy',
            'UNRATE': 'unemployment',
            'T10Y2Y': 't10y2y',
        }

        for col_name in macro_cols.values():
            df[col_name] = np.nan

        if self.db_manager is None:
            return df

        try:
            start_date = df.index.min().date() if hasattr(df.index.min(), 'date') else df.index.min()
            end_date = df.index.max().date() if hasattr(df.index.max(), 'date') else df.index.max()

            indicators = self.db_manager.get_macro_indicators(
                list(macro_cols.keys()), start_date, end_date
            )

            if not indicators:
                return df

            # Build a DataFrame from macro data
            for indicator in indicators:
                name = indicator['indicator_name']
                col_name = macro_cols.get(name)
                if col_name:
                    date = pd.Timestamp(indicator['date'])
                    if date in df.index:
                        df.loc[date, col_name] = indicator['value']

            # Forward-fill macro data (released less frequently)
            for col_name in macro_cols.values():
                df[col_name] = df[col_name].ffill()

        except Exception as e:
            logger.warning(f"Failed to add macro features: {e}")

        return df
