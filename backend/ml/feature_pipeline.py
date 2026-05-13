"""
Feature engineering pipeline for AI Stock GPT v2.
Computes technical, volatility, momentum, mean-reversion, regime,
cross-asset, and macro indicators from price data.

v2 improvements over v1:
- Mean-reversion z-scores
- VWAP approximation
- Price acceleration
- Relative volume
- EMA crossover signals
- Gap features
- Volatility regime classification
- Cyclical time encoding
- Feature normalization for ensemble
- Removed duplicate features (rolling_mean_20 == sma_20)
- Lag features use returns, not raw prices
- Vectorized OBV
"""

import logging
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Unified feature engineering: technical, volatility, momentum, mean-reversion, regime, macro."""

    _YF_SUFFIX_MAP = {'.BSE': '.BO', '.NSE': '.NS'}

    def __init__(self, db_manager=None, av_collector=None):
        self.db_manager = db_manager
        self.av_collector = av_collector

    @staticmethod
    def _yf_symbol(symbol: str) -> str:
        for av_suffix, yf_suffix in FeaturePipeline._YF_SUFFIX_MAP.items():
            if symbol.upper().endswith(av_suffix):
                return symbol[: -len(av_suffix)] + yf_suffix
        return symbol

    @staticmethod
    def _is_indian(symbol: str) -> bool:
        upper = symbol.upper()
        return upper.endswith('.BSE') or upper.endswith('.BO') or upper.endswith('.NS') or upper.endswith('.NSE')

    # ── public API ──────────────────────────────────────────────

    def build_features(self, symbol: str, lookback_days: int = 504) -> pd.DataFrame:
        """Fetch price data, compute all features, return clean DataFrame."""
        df = self._fetch_price_data(symbol, lookback_days)

        if df.empty:
            raise ValueError(f"No price data available for {symbol}")

        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        # Core feature groups
        df = self._add_technical_indicators(df)
        df = self._add_volatility_indicators(df)
        df = self._add_momentum_indicators(df, symbol)
        df = self._add_mean_reversion_features(df)
        df = self._add_volume_features(df)
        df = self._add_gap_features(df)
        df = self._add_crossover_signals(df)
        df = self._add_regime_features(df)
        df = self._add_lag_features(df)
        df = self._add_time_features(df)
        df = self._add_macro_features(df)

        # Drop all-NaN columns
        all_nan_cols = [c for c in df.columns if df[c].isna().all()]
        if all_nan_cols:
            df = df.drop(columns=all_nan_cols)
            logger.info(f"Dropped {len(all_nan_cols)} all-NaN columns")

        # Replace infinities and drop NaN rows
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()

        logger.info(f"Built features for {symbol}: {df.shape}")
        return df

    def compute_indicators(self, symbol: str) -> Dict[str, Any]:
        """Compute latest indicator values for chat display."""
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
                'vol_regime': float(latest.get('vol_regime', 0)),
                'trend_regime': float(latest.get('trend_regime', 0)),
            }

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

    # ── data fetching ───────────────────────────────────────────

    def _fetch_price_data(self, symbol: str, lookback_days: int) -> pd.DataFrame:
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
            yf_sym = self._yf_symbol(symbol)
            try:
                ticker = yf.Ticker(yf_sym)
                df = ticker.history(period=f"{lookback_days}d")
                logger.info(f"Feature data source for {symbol}: yfinance ({yf_sym})")
            except Exception as e:
                logger.warning(f"yfinance fetch failed for {symbol}: {e}")

        return df

    # ── technical indicators ────────────────────────────────────

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # Moving averages
        df['sma_20'] = close.rolling(20).mean()
        df['sma_50'] = close.rolling(50).mean()
        df['sma_200'] = close.rolling(200).mean()
        df['ema_12'] = close.ewm(span=12, adjust=False).mean()
        df['ema_26'] = close.ewm(span=26, adjust=False).mean()
        df['ema_50'] = close.ewm(span=50, adjust=False).mean()

        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['bb_middle'] = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        bb_width = df['bb_upper'] - df['bb_lower']
        df['bb_width_pct'] = bb_width / df['bb_middle'].replace(0, np.nan)
        df['bb_position'] = (close - df['bb_lower']) / bb_width.replace(0, np.nan)

        # ATR
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()
        df['atr_ratio'] = df['atr_14'] / close.replace(0, np.nan)

        # OBV — vectorized
        direction = np.sign(close.diff())
        df['obv'] = (direction * volume).fillna(0).cumsum()

        # Stochastic
        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        df['stoch_k'] = 100 * (close - low_14) / (high_14 - low_14).replace(0, np.nan)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        # ADX
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14.replace(0, np.nan))
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14.replace(0, np.nan))
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        df['adx_14'] = dx.rolling(14).mean()

        # VWAP approximation (intraday proxy using daily typical price * volume)
        typical_price = (high + low + close) / 3
        cum_tp_vol = (typical_price * volume).rolling(20).sum()
        cum_vol = volume.rolling(20).sum()
        df['vwap_20'] = cum_tp_vol / cum_vol.replace(0, np.nan)
        df['price_vs_vwap'] = (close - df['vwap_20']) / df['vwap_20'].replace(0, np.nan)

        return df

    # ── volatility ──────────────────────────────────────────────

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df['close']
        returns = close.pct_change()

        df['hist_vol_10'] = returns.rolling(10).std() * np.sqrt(252)
        df['hist_vol_20'] = returns.rolling(20).std() * np.sqrt(252)
        df['hist_vol_60'] = returns.rolling(60).std() * np.sqrt(252)

        # Garman-Klass volatility
        log_hl = np.log(df['high'] / df['low'].replace(0, np.nan)) ** 2
        log_co = np.log(df['close'] / df['open'].replace(0, np.nan)) ** 2
        df['garman_klass'] = np.sqrt(
            (0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(20).mean() * 252
        )

        # Volatility of volatility
        df['vol_of_vol'] = df['hist_vol_20'].rolling(20).std()

        # Volatility ratio (short vs long) — detects volatility expansion/contraction
        df['vol_ratio'] = df['hist_vol_10'] / df['hist_vol_60'].replace(0, np.nan)

        # Realized vs expected volatility (ATR-based)
        if 'atr_14' in df.columns:
            df['realized_vs_atr'] = df['hist_vol_20'] / (df['atr_ratio'].replace(0, np.nan) * np.sqrt(252))

        return df

    # ── momentum ────────────────────────────────────────────────

    def _add_momentum_indicators(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        close = df['close']
        returns = close.pct_change()

        # Multi-period returns
        for period in [1, 5, 10, 21, 63, 126, 252]:
            df[f'return_{period}d'] = close.pct_change(period)

        # Price acceleration (change in momentum)
        df['momentum_5d'] = df['return_5d']
        df['momentum_21d'] = df['return_21d']
        df['price_acceleration'] = df['return_5d'] - df['return_5d'].shift(5)

        # Rate of change (normalized)
        df['roc_10'] = close.pct_change(10)
        df['roc_21'] = close.pct_change(21)

        # Beta vs benchmark
        benchmark_sym = '^BSESN' if self._is_indian(symbol) else 'SPY'
        try:
            bench = yf.Ticker(benchmark_sym).history(period=f"{len(df) + 30}d")
            if not bench.empty:
                bench.columns = [c.lower().replace(' ', '_') for c in bench.columns]
                bench_returns = bench['close'].pct_change()
                aligned = pd.DataFrame({
                    'stock': returns,
                    'bench': bench_returns
                }).dropna()
                if len(aligned) > 60:
                    rolling_cov = aligned['stock'].rolling(60).cov(aligned['bench'])
                    rolling_var = aligned['bench'].rolling(60).var()
                    beta = rolling_cov / rolling_var.replace(0, np.nan)
                    df['beta_spy'] = beta.reindex(df.index)

                    # Relative strength vs benchmark
                    stock_cum = (1 + aligned['stock']).rolling(21).apply(lambda x: x.prod(), raw=True)
                    bench_cum = (1 + aligned['bench']).rolling(21).apply(lambda x: x.prod(), raw=True)
                    df['relative_strength_21'] = (stock_cum / bench_cum.replace(0, np.nan)).reindex(df.index)
                else:
                    df['beta_spy'] = np.nan
                    df['relative_strength_21'] = np.nan
            else:
                df['beta_spy'] = np.nan
                df['relative_strength_21'] = np.nan
        except Exception:
            df['beta_spy'] = np.nan
            df['relative_strength_21'] = np.nan

        # Rolling Sharpe (21-day)
        df['rolling_sharpe_21'] = (
            returns.rolling(21).mean() / returns.rolling(21).std().replace(0, np.nan)
        ) * np.sqrt(252)

        # Sortino ratio (21-day, downside only)
        downside = returns.copy()
        downside[downside > 0] = 0
        df['rolling_sortino_21'] = (
            returns.rolling(21).mean() / downside.rolling(21).std().replace(0, np.nan)
        ) * np.sqrt(252)

        return df

    # ── mean-reversion features ─────────────────────────────────

    def _add_mean_reversion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Z-scores and distance from moving averages — high-signal for classification."""
        close = df['close']

        # Z-score: how many std devs from 20-day mean
        rolling_mean = close.rolling(20).mean()
        rolling_std = close.rolling(20).std()
        df['zscore_20'] = (close - rolling_mean) / rolling_std.replace(0, np.nan)
        df['zscore_50'] = (close - close.rolling(50).mean()) / close.rolling(50).std().replace(0, np.nan)

        # Distance from SMAs (percentage)
        df['dist_sma_20'] = (close - df['sma_20']) / df['sma_20'].replace(0, np.nan)
        df['dist_sma_50'] = (close - df['sma_50']) / df['sma_50'].replace(0, np.nan)
        if 'sma_200' in df.columns:
            df['dist_sma_200'] = (close - df['sma_200']) / df['sma_200'].replace(0, np.nan)

        # RSI mean reversion signal
        if 'rsi_14' in df.columns:
            df['rsi_zscore'] = (df['rsi_14'] - df['rsi_14'].rolling(63).mean()) / df['rsi_14'].rolling(63).std().replace(0, np.nan)

        # Return z-score (is the recent return extreme?)
        ret_5 = close.pct_change(5)
        df['return_zscore_5d'] = (ret_5 - ret_5.rolling(63).mean()) / ret_5.rolling(63).std().replace(0, np.nan)

        return df

    # ── volume features ─────────────────────────────────────────

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        volume = df['volume']

        # Relative volume (vs 20-day average)
        vol_ma = volume.rolling(20).mean()
        df['relative_volume'] = volume / vol_ma.replace(0, np.nan)

        # Volume momentum
        df['volume_change_5d'] = volume.pct_change(5)

        # Volume-price divergence: price up but volume down (bearish divergence)
        price_dir = np.sign(df['close'].pct_change(5))
        vol_dir = np.sign(volume.pct_change(5))
        df['vol_price_divergence'] = (price_dir != vol_dir).astype(float)

        # Volume trend
        df['volume_sma_ratio'] = volume / volume.rolling(50).mean().replace(0, np.nan)

        return df

    # ── gap features ────────────────────────────────────────────

    def _add_gap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gap up/down at open — important for momentum."""
        df['gap_pct'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1).replace(0, np.nan)
        df['gap_filled'] = (
            ((df['gap_pct'] > 0) & (df['low'] <= df['close'].shift(1))) |
            ((df['gap_pct'] < 0) & (df['high'] >= df['close'].shift(1)))
        ).astype(float)
        df['avg_gap_5d'] = df['gap_pct'].rolling(5).mean()

        return df

    # ── crossover signals ───────────────────────────────────────

    def _add_crossover_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """EMA/SMA crossover signals — binary features."""
        # Golden cross / death cross signals
        df['sma_20_50_cross'] = np.where(df['sma_20'] > df['sma_50'], 1, -1).astype(float)
        df['ema_12_26_cross'] = np.where(df['ema_12'] > df['ema_26'], 1, -1).astype(float)

        # Price vs key MAs
        df['price_above_sma50'] = (df['close'] > df['sma_50']).astype(float)
        df['price_above_sma200'] = (df['close'] > df['sma_200']).astype(float) if 'sma_200' in df.columns else 0.0

        # MACD crossover
        df['macd_cross_signal'] = np.where(df['macd'] > df['macd_signal'], 1, -1).astype(float)

        # Stochastic crossover
        if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
            df['stoch_cross'] = np.where(df['stoch_k'] > df['stoch_d'], 1, -1).astype(float)

        return df

    # ── regime features ─────────────────────────────────────────

    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market regime classification features."""
        close = df['close']
        returns = close.pct_change()

        # Volatility regime (0=low, 1=medium, 2=high)
        vol_20 = returns.rolling(20).std() * np.sqrt(252)
        vol_percentile = vol_20.rolling(252, min_periods=63).rank(pct=True)
        df['vol_regime'] = pd.cut(vol_percentile, bins=[0, 0.33, 0.66, 1.0],
                                   labels=[0, 1, 2], include_lowest=True).astype(float)

        # Trend regime using SMA slope
        sma_50_slope = df['sma_50'].pct_change(10) if 'sma_50' in df.columns else close.rolling(50).mean().pct_change(10)
        df['trend_regime'] = pd.cut(sma_50_slope,
                                     bins=[-np.inf, -0.01, 0.01, np.inf],
                                     labels=[-1, 0, 1]).astype(float)

        # Consecutive up/down days
        daily_dir = np.sign(returns)
        streaks = daily_dir.groupby((daily_dir != daily_dir.shift()).cumsum()).cumcount() + 1
        df['streak_days'] = streaks * daily_dir

        # Drawdown from rolling high
        rolling_max = close.rolling(63, min_periods=1).max()
        df['drawdown_63d'] = (close - rolling_max) / rolling_max.replace(0, np.nan)

        # Market breadth proxy: how many of the last 21 days were positive?
        df['pct_positive_days_21'] = (returns > 0).rolling(21).mean()

        return df

    # ── lag features ────────────────────────────────────────────

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lag features using RETURNS (not raw prices) to avoid scale leakage."""
        returns = df['close'].pct_change()
        vol_returns = df['volume'].pct_change()

        df['return_lag_1'] = returns.shift(1)
        df['return_lag_2'] = returns.shift(2)
        df['return_lag_5'] = returns.shift(5)
        df['return_lag_10'] = returns.shift(10)
        df['vol_return_lag_1'] = vol_returns.shift(1)
        df['vol_return_lag_5'] = vol_returns.shift(5)

        # Auto-correlation of returns
        df['return_autocorr_5'] = returns.rolling(21).apply(
            lambda x: x.autocorr(lag=5) if len(x) > 5 else 0, raw=False
        )

        return df

    # ── time features ───────────────────────────────────────────

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cyclical time encoding (sin/cos) instead of raw integers."""
        idx = df.index
        dow = idx.dayofweek
        month = idx.month

        # Cyclical encoding
        df['day_sin'] = np.sin(2 * np.pi * dow / 5)
        df['day_cos'] = np.cos(2 * np.pi * dow / 5)
        df['month_sin'] = np.sin(2 * np.pi * month / 12)
        df['month_cos'] = np.cos(2 * np.pi * month / 12)
        df['quarter'] = idx.quarter

        return df

    # ── macro features ──────────────────────────────────────────

    def _add_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
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

            for indicator in indicators:
                name = indicator['indicator_name']
                col_name = macro_cols.get(name)
                if col_name:
                    date = pd.Timestamp(indicator['date'])
                    if date in df.index:
                        df.loc[date, col_name] = indicator['value']

            for col_name in macro_cols.values():
                df[col_name] = df[col_name].ffill()

        except Exception as e:
            logger.warning(f"Failed to add macro features: {e}")

        return df
