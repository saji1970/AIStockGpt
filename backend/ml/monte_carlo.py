"""
Monte Carlo simulation using Geometric Brownian Motion with correlated returns.
"""

import logging
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """Monte Carlo simulation using Geometric Brownian Motion with correlated returns."""

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
        """Estimate annualized mu and covariance from historical yfinance data."""
        prices = {}
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{lookback_days + 30}d")
            if hist.empty:
                raise ValueError(f"No price data for {symbol}")
            prices[symbol] = hist['Close']

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
