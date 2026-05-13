"""
Monte Carlo simulation using Geometric Brownian Motion with correlated returns.
Alpha Vantage is the primary data source; yfinance is the fallback.
"""

import logging
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """Monte Carlo simulation using Geometric Brownian Motion with correlated returns."""

    def __init__(self, av_collector=None):
        self.av_collector = av_collector

    def simulate(
        self,
        symbols: List[str],
        weights: List[float],
        initial_amount: float,
        months: int,
        n_simulations: int = 10000,
    ) -> Dict[str, Any]:
        """
        Simulate portfolio value over time.
        Uses historical returns to estimate mu (drift) and covariance matrix.
        Cholesky decomposition for correlated multi-asset simulation.
        """
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize

        days = int(months * 21)  # Trading days

        mu, cov = self._estimate_parameters(symbols)
        paths = self._gbm_paths(mu, cov, weights, initial_amount, days, n_simulations)

        final_values = paths[:, -1]

        # Compute metrics
        percentiles = {
            5: float(np.percentile(final_values, 5)),
            25: float(np.percentile(final_values, 25)),
            50: float(np.percentile(final_values, 50)),
            75: float(np.percentile(final_values, 75)),
            95: float(np.percentile(final_values, 95)),
        }

        returns = (final_values - initial_amount) / initial_amount
        annual_factor = 12 / months if months > 0 else 1

        # Value at Risk
        losses = initial_amount - final_values
        var_95 = float(np.percentile(losses, 95))
        cvar_95 = float(np.mean(losses[losses >= var_95])) if (losses >= var_95).any() else var_95

        return {
            'initial_amount': initial_amount,
            'months': months,
            'median_value': float(np.median(final_values)),
            'mean_value': float(np.mean(final_values)),
            'percentiles': percentiles,
            'probability_positive': float(np.mean(returns > 0)),
            'probability_double': float(np.mean(final_values >= 2 * initial_amount)),
            'expected_annual_return': float(np.mean(returns) * annual_factor),
            'var_95': var_95,
            'cvar_95': cvar_95,
            'best_case': percentiles[95],
            'worst_case': percentiles[5],
        }

    def _estimate_parameters(
        self, symbols: List[str], lookback_days: int = 252
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate annualized mu and covariance from historical data."""
        prices = {}

        # Try Alpha Vantage first
        if self.av_collector:
            for symbol in symbols:
                try:
                    df = self.av_collector.get_daily_history(symbol, days=lookback_days + 30)
                    if df is not None and not df.empty and 'close' in df.columns:
                        prices[symbol] = df['close']
                except Exception as e:
                    logger.warning(f"Alpha Vantage history failed for {symbol}: {e}")

        # Fallback to yfinance for missing symbols
        missing = [s for s in symbols if s not in prices]
        if missing:
            try:
                import yfinance as yf
                for symbol in missing:
                    try:
                        from backend.ml.portfolio_optimizer import _to_yf_symbol
                    except ImportError:
                        _to_yf_symbol = lambda s: s
                    ticker = yf.Ticker(_to_yf_symbol(symbol))
                    hist = ticker.history(period=f"{lookback_days + 30}d")
                    if not hist.empty:
                        prices[symbol] = hist['Close']
            except Exception as e:
                logger.warning(f"yfinance fallback failed: {e}")

        if not prices:
            raise ValueError("No price data available for any symbol")

        price_df = pd.DataFrame(prices).dropna()

        if len(price_df) < 30:
            raise ValueError("Insufficient price data for parameter estimation")

        returns = price_df.pct_change().dropna()

        # Annualized drift (mu)
        mu = returns.mean().values * 252

        # Annualized covariance
        cov = returns.cov().values * 252

        return mu, cov

    def _gbm_paths(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        weights: np.ndarray,
        initial: float,
        days: int,
        n_sims: int,
    ) -> np.ndarray:
        """Generate correlated GBM paths. Returns array of shape (n_sims, days)."""
        n_assets = len(mu)

        # Portfolio drift and volatility
        port_mu = np.dot(weights, mu)
        port_var = np.dot(weights, np.dot(cov, weights))
        port_vol = np.sqrt(port_var)

        # Daily parameters
        dt = 1 / 252
        drift = (port_mu - 0.5 * port_var) * dt
        diffusion = port_vol * np.sqrt(dt)

        # Generate random paths
        rng = np.random.default_rng(seed=42)
        Z = rng.standard_normal((n_sims, days))

        # Build paths
        log_returns = drift + diffusion * Z
        log_paths = np.cumsum(log_returns, axis=1)

        # Prepend initial value
        paths = initial * np.exp(
            np.column_stack([np.zeros(n_sims), log_paths])
        )

        return paths
