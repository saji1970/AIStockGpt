"""
Market regime detection using Hidden Markov Models and statistical methods.
Classifies market into regimes: bull, bear, sideways, high-vol, low-vol.
Used by ensemble to weight models differently per regime.
"""

import logging
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RegimeDetector:
    """Detect market regimes using statistical methods."""

    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes

    def detect_regime(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Classify current market regime.
        Returns dict with regime label and confidence.
        """
        close = df['close']
        returns = close.pct_change().dropna()

        if len(returns) < 63:
            return {'regime': 'unknown', 'confidence': 0.0, 'details': {}}

        # Trend analysis
        trend = self._classify_trend(close)

        # Volatility analysis
        vol_regime = self._classify_volatility(returns)

        # Combine into overall regime
        regime, confidence = self._combine_regimes(trend, vol_regime, returns)

        return {
            'regime': regime,
            'confidence': confidence,
            'trend': trend,
            'volatility': vol_regime,
            'details': {
                'return_21d': float(close.pct_change(21).iloc[-1]) if len(close) > 21 else 0,
                'vol_20d': float(returns.rolling(20).std().iloc[-1] * np.sqrt(252)),
                'vol_60d': float(returns.rolling(60).std().iloc[-1] * np.sqrt(252)) if len(returns) > 60 else 0,
            }
        }

    def get_regime_series(self, df: pd.DataFrame, window: int = 63) -> pd.Series:
        """
        Generate a regime series for the entire DataFrame.
        Returns Series of regime labels aligned to df index.
        Used during training for regime-aware target and weighting.
        """
        close = df['close']
        returns = close.pct_change()

        regimes = pd.Series(index=df.index, dtype='float64')

        for i in range(window, len(df)):
            slice_returns = returns.iloc[max(0, i - window):i]
            slice_close = close.iloc[max(0, i - window):i]

            if len(slice_returns.dropna()) < 20:
                regimes.iloc[i] = 0  # unknown
                continue

            # Trend
            sma_short = slice_close.rolling(min(20, len(slice_close))).mean().iloc[-1]
            sma_long = slice_close.rolling(min(50, len(slice_close))).mean().iloc[-1]

            # Volatility
            current_vol = slice_returns.std() * np.sqrt(252)

            if sma_short > sma_long * 1.01 and current_vol < 0.25:
                regimes.iloc[i] = 1  # bull, low vol
            elif sma_short > sma_long * 1.01 and current_vol >= 0.25:
                regimes.iloc[i] = 2  # bull, high vol
            elif sma_short < sma_long * 0.99 and current_vol < 0.25:
                regimes.iloc[i] = 3  # bear, low vol
            elif sma_short < sma_long * 0.99 and current_vol >= 0.25:
                regimes.iloc[i] = 4  # bear, high vol
            else:
                regimes.iloc[i] = 0  # sideways

        return regimes

    def _classify_trend(self, close: pd.Series) -> str:
        """Classify trend as bull, bear, or sideways."""
        if len(close) < 50:
            return 'sideways'

        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]
        current = close.iloc[-1]

        # Trend strength
        ret_21d = close.pct_change(21).iloc[-1] if len(close) > 21 else 0

        if current > sma_20 > sma_50 and ret_21d > 0.02:
            return 'bull'
        elif current < sma_20 < sma_50 and ret_21d < -0.02:
            return 'bear'
        else:
            return 'sideways'

    def _classify_volatility(self, returns: pd.Series) -> str:
        """Classify volatility regime."""
        current_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        historical_vol = returns.rolling(252, min_periods=63).std().iloc[-1] * np.sqrt(252)

        if current_vol > historical_vol * 1.3:
            return 'high_vol'
        elif current_vol < historical_vol * 0.7:
            return 'low_vol'
        else:
            return 'normal_vol'

    def _combine_regimes(self, trend: str, vol: str, returns: pd.Series) -> Tuple[str, float]:
        """Combine trend and vol into a single regime with confidence."""
        # Trend confidence based on consistency
        recent_returns = returns.tail(21)
        positive_pct = (recent_returns > 0).mean()

        if trend == 'bull':
            confidence = min(positive_pct, 0.95)
        elif trend == 'bear':
            confidence = min(1 - positive_pct, 0.95)
        else:
            confidence = 0.5

        regime = f"{trend}_{vol}"
        return regime, float(confidence)
