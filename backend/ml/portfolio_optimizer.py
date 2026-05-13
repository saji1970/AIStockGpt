"""
Portfolio optimization using PyPortfolioOpt.
"""

import logging
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from pypfopt import expected_returns, risk_models, EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

logger = logging.getLogger(__name__)

# Mapping for yfinance: Alpha Vantage suffix -> yfinance suffix
_YF_SUFFIX_MAP = {'.BSE': '.BO', '.NSE': '.NS'}


def _to_yf_symbol(symbol: str) -> str:
    """Convert an Alpha Vantage symbol to its yfinance equivalent."""
    for av_suffix, yf_suffix in _YF_SUFFIX_MAP.items():
        if symbol.upper().endswith(av_suffix):
            return symbol[: -len(av_suffix)] + yf_suffix
    return symbol


# Symbol pools by risk level and market
SYMBOL_POOLS_US = {
    'conservative': [
        'BND', 'AGG', 'TLT', 'VTI', 'GLD', 'VIG', 'SCHD', 'VYM', 'JNJ', 'PG',
    ],
    'moderate': [
        'VTI', 'QQQ', 'BND', 'GLD', 'VIG', 'AAPL', 'MSFT', 'GOOGL', 'JNJ', 'PG',
    ],
    'aggressive': [
        'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'AMD', 'CRM',
    ],
}

SYMBOL_POOLS_INDIA = {
    'conservative': [
        'HDFCBANK.BSE', 'SBIN.BSE', 'ITC.BSE', 'HINDUNILVR.BSE',
        'NESTLEIND.BSE', 'BRITANNIA.BSE', 'POWERGRID.BSE', 'NTPC.BSE',
    ],
    'moderate': [
        'HDFCBANK.BSE', 'ICICIBANK.BSE', 'INFY.BSE', 'TCS.BSE',
        'RELIANCE.BSE', 'HINDUNILVR.BSE', 'ITC.BSE', 'LT.BSE',
        'KOTAKBANK.BSE', 'SBIN.BSE',
    ],
    'aggressive': [
        'INFY.BSE', 'TATAMOTORS.BSE', 'BAJFINANCE.BSE', 'ADANIENT.BSE',
        'RELIANCE.BSE', 'ICICIBANK.BSE', 'HCLTECH.BSE', 'MARUTI.BSE',
        'TITAN.BSE', 'BHARTIARTL.BSE',
    ],
}

# Mixed (global) pool - default when no market specified
SYMBOL_POOLS = {
    'conservative': [
        'BND', 'AGG', 'TLT', 'VTI', 'GLD', 'VIG', 'SCHD',
        'HDFCBANK.BSE', 'SBIN.BSE', 'ITC.BSE',
    ],
    'moderate': [
        'VTI', 'QQQ', 'BND', 'GLD', 'VIG', 'AAPL', 'MSFT', 'GOOGL', 'JNJ', 'PG',
        'INFY.BSE', 'TCS.BSE', 'HDFCBANK.BSE', 'RELIANCE.BSE',
    ],
    'aggressive': [
        'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'AMD', 'CRM',
        'INFY.BSE', 'TATAMOTORS.BSE', 'BAJFINANCE.BSE', 'ADANIENT.BSE',
    ],
}


class PortfolioOptimizer:
    """Portfolio optimization using PyPortfolioOpt."""

    def optimize(
        self,
        symbols: List[str],
        method: str = 'max_sharpe',
        constraints: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Optimize portfolio allocation.
        Methods: 'max_sharpe', 'min_volatility', 'risk_parity'
        """
        prices = self._get_price_data(symbols)

        if prices.empty or len(prices) < 60:
            raise ValueError("Insufficient price data for optimization")

        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.sample_cov(prices)

        if method == 'risk_parity':
            # Equal risk contribution approximation
            weights = self._risk_parity(S)
        else:
            ef = EfficientFrontier(mu, S)

            # Apply constraints if provided
            if constraints:
                if 'weight_bounds' in constraints:
                    ef = EfficientFrontier(mu, S, weight_bounds=constraints['weight_bounds'])

            if method == 'max_sharpe':
                ef.max_sharpe()
            elif method == 'min_volatility':
                ef.min_volatility()
            else:
                ef.max_sharpe()  # Default

            weights = ef.clean_weights()

        # Compute performance metrics
        weights_array = np.array([weights.get(s, 0) for s in symbols])
        returns = prices.pct_change().dropna()
        port_returns = returns.dot(weights_array)

        annual_return = float(port_returns.mean() * 252)
        annual_vol = float(port_returns.std() * np.sqrt(252))
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        return {
            'weights': {s: float(w) for s, w in weights.items() if w > 0.001},
            'expected_annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'method': method,
            'symbols': symbols,
        }

    def risk_analysis(
        self,
        symbols: List[str],
        weights: List[float],
    ) -> Dict[str, Any]:
        """Compute risk metrics for a given allocation."""
        prices = self._get_price_data(symbols)

        if prices.empty:
            raise ValueError("Insufficient price data for risk analysis")

        weights = np.array(weights)
        weights = weights / weights.sum()

        returns = prices.pct_change().dropna()
        port_returns = returns.dot(weights)

        # VaR and CVaR (daily)
        var_95 = float(-np.percentile(port_returns, 5))
        cvar_95_mask = port_returns <= -var_95
        cvar_95 = float(-port_returns[cvar_95_mask].mean()) if cvar_95_mask.any() else var_95

        # Max drawdown
        cumulative = (1 + port_returns).cumprod()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative - peak) / peak
        max_drawdown = float(drawdown.min())

        # Sharpe ratio
        annual_return = float(port_returns.mean() * 252)
        annual_vol = float(port_returns.std() * np.sqrt(252))
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        # Beta vs SPY
        beta = self._compute_beta(port_returns)

        # Correlation matrix
        corr = returns.corr()
        correlation_matrix = {
            s1: {s2: float(corr.loc[s1, s2]) for s2 in symbols if s2 in corr.columns}
            for s1 in symbols if s1 in corr.index
        }

        return {
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'annual_volatility': annual_vol,
            'beta': beta,
            'correlation_matrix': correlation_matrix,
        }

    def recommend_allocation(
        self,
        amount: float,
        risk_level: str,
        horizon_months: int,
        market: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a recommended allocation based on risk profile and market.

        Args:
            amount: Investment amount
            risk_level: 'conservative', 'moderate', or 'aggressive'
            horizon_months: Investment horizon in months
            market: 'india', 'us', or None (global/mixed)
        """
        risk_level = risk_level.lower()
        if risk_level not in SYMBOL_POOLS:
            risk_level = 'moderate'

        # Select pool based on target market
        if market == 'india':
            pool = SYMBOL_POOLS_INDIA
        elif market == 'us':
            pool = SYMBOL_POOLS_US
        else:
            pool = SYMBOL_POOLS

        symbols = pool[risk_level]

        # Choose optimization method based on risk level
        method_map = {
            'conservative': 'min_volatility',
            'moderate': 'max_sharpe',
            'aggressive': 'max_sharpe',
        }
        method = method_map[risk_level]

        # Filter to symbols with available data
        valid_symbols = []
        for s in symbols:
            try:
                ticker = yf.Ticker(_to_yf_symbol(s))
                hist = ticker.history(period="5d")
                if not hist.empty:
                    valid_symbols.append(s)
            except Exception:
                continue

        if len(valid_symbols) < 2:
            raise ValueError("Insufficient valid symbols for optimization")

        # Optimize
        result = self.optimize(valid_symbols, method)
        weights = result['weights']

        # Compute allocations
        allocations = {}
        for symbol, weight in weights.items():
            sym_amount = amount * weight
            try:
                ticker = yf.Ticker(_to_yf_symbol(symbol))
                latest_price = ticker.history(period="1d")['Close'].iloc[-1]
                shares_approx = sym_amount / latest_price
            except Exception:
                shares_approx = 0

            allocations[symbol] = {
                'weight': weight,
                'amount': round(sym_amount, 2),
                'shares_approx': round(shares_approx, 2),
            }

        # Risk metrics on the recommended allocation
        risk_metrics = self.risk_analysis(
            list(weights.keys()),
            list(weights.values())
        )

        # Expected return ranges
        annual_ret = result['expected_annual_return']
        annual_vol = result['annual_volatility']

        # Determine currency from the market
        currency = 'INR' if market == 'india' else 'USD'
        currency_symbol = '₹' if market == 'india' else '$'

        return {
            'allocations': allocations,
            'risk_level': risk_level,
            'market': market or 'global',
            'currency': currency,
            'currency_symbol': currency_symbol,
            'expected_return_range': {
                'low': annual_ret - annual_vol,
                'mid': annual_ret,
                'high': annual_ret + annual_vol,
            },
            'risk_metrics': risk_metrics,
            'method': method,
        }

    def _get_price_data(self, symbols: List[str], period_days: int = 504) -> pd.DataFrame:
        """Fetch adjusted close prices via yfinance."""
        prices = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(_to_yf_symbol(symbol))
                hist = ticker.history(period=f"{period_days}d")
                if not hist.empty:
                    prices[symbol] = hist['Close']
            except Exception as e:
                logger.warning(f"Failed to fetch prices for {symbol}: {e}")

        if not prices:
            return pd.DataFrame()

        return pd.DataFrame(prices).dropna()

    def _risk_parity(self, cov_matrix: pd.DataFrame) -> Dict[str, float]:
        """Simple risk parity: equal risk contribution."""
        symbols = cov_matrix.columns.tolist()
        cov = cov_matrix.values
        n = len(symbols)

        # Inverse volatility weighting as approximation
        vols = np.sqrt(np.diag(cov))
        inv_vols = 1.0 / vols
        weights = inv_vols / inv_vols.sum()

        return {s: float(w) for s, w in zip(symbols, weights)}

    def _compute_beta(self, port_returns: pd.Series) -> float:
        """Compute beta vs SPY."""
        try:
            spy = yf.Ticker('SPY').history(period=f"{len(port_returns) + 30}d")
            if spy.empty:
                return 1.0
            spy_returns = spy['Close'].pct_change().dropna()

            aligned = pd.DataFrame({
                'port': port_returns,
                'spy': spy_returns
            }).dropna()

            if len(aligned) < 30:
                return 1.0

            cov = aligned['port'].cov(aligned['spy'])
            var = aligned['spy'].var()
            return float(cov / var) if var > 0 else 1.0
        except Exception:
            return 1.0
