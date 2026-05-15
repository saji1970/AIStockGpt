"""
Portfolio optimization using PyPortfolioOpt.
Alpha Vantage is the primary data source; yfinance is the fallback.
"""

import logging
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
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

SYMBOL_POOLS_CRYPTO = {
    'conservative': [
        'BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD',
    ],
    'moderate': [
        'BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD',
    ],
    'aggressive': [
        'BTC-USD', 'ETH-USD', 'SOL-USD', 'AVAX-USD', 'DOT-USD', 'LINK-USD',
        'DOGE-USD', 'XRP-USD', 'ADA-USD', 'BNB-USD',
    ],
}

SYMBOL_POOLS_COMMODITY = {
    'conservative': [
        'GC=F', 'SI=F', 'GLD', 'BND',
    ],
    'moderate': [
        'GC=F', 'SI=F', 'CL=F', 'HG=F', 'GLD',
    ],
    'aggressive': [
        'GC=F', 'SI=F', 'CL=F', 'NG=F', 'HG=F', 'PL=F',
    ],
}

# Default pool when no market specified - US-only (most common default)
SYMBOL_POOLS = {
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

# Static fallback allocations when live data is completely unavailable
_STATIC_ALLOCATIONS = {
    'us': {
        'conservative': {'BND': 0.30, 'AGG': 0.20, 'VTI': 0.20, 'GLD': 0.15, 'VIG': 0.15},
        'moderate':     {'VTI': 0.25, 'QQQ': 0.20, 'BND': 0.15, 'AAPL': 0.15, 'MSFT': 0.15, 'GLD': 0.10},
        'aggressive':   {'QQQ': 0.20, 'AAPL': 0.15, 'MSFT': 0.15, 'NVDA': 0.15, 'GOOGL': 0.15, 'AMZN': 0.10, 'TSLA': 0.10},
    },
    'india': {
        'conservative': {'HDFCBANK.BSE': 0.25, 'SBIN.BSE': 0.20, 'ITC.BSE': 0.20, 'HINDUNILVR.BSE': 0.20, 'POWERGRID.BSE': 0.15},
        'moderate':     {'HDFCBANK.BSE': 0.15, 'INFY.BSE': 0.15, 'TCS.BSE': 0.15, 'RELIANCE.BSE': 0.20, 'ICICIBANK.BSE': 0.15, 'ITC.BSE': 0.10, 'LT.BSE': 0.10},
        'aggressive':   {'INFY.BSE': 0.15, 'TATAMOTORS.BSE': 0.15, 'BAJFINANCE.BSE': 0.15, 'RELIANCE.BSE': 0.20, 'ADANIENT.BSE': 0.15, 'HCLTECH.BSE': 0.10, 'TITAN.BSE': 0.10},
    },
    'global': {
        'conservative': {'BND': 0.25, 'AGG': 0.15, 'VTI': 0.20, 'GLD': 0.15, 'VIG': 0.15, 'SCHD': 0.10},
        'moderate':     {'VTI': 0.20, 'QQQ': 0.15, 'AAPL': 0.15, 'MSFT': 0.15, 'BND': 0.15, 'GLD': 0.10, 'GOOGL': 0.10},
        'aggressive':   {'QQQ': 0.20, 'AAPL': 0.15, 'NVDA': 0.15, 'MSFT': 0.15, 'GOOGL': 0.10, 'AMZN': 0.10, 'TSLA': 0.10, 'META': 0.05},
    },
    'crypto': {
        'conservative': {'BTC-USD': 0.50, 'ETH-USD': 0.30, 'BNB-USD': 0.10, 'SOL-USD': 0.10},
        'moderate':     {'BTC-USD': 0.35, 'ETH-USD': 0.25, 'SOL-USD': 0.15, 'XRP-USD': 0.10, 'ADA-USD': 0.10, 'BNB-USD': 0.05},
        'aggressive':   {'BTC-USD': 0.25, 'ETH-USD': 0.20, 'SOL-USD': 0.15, 'AVAX-USD': 0.10, 'DOT-USD': 0.10, 'LINK-USD': 0.10, 'DOGE-USD': 0.10},
    },
    'commodity': {
        'conservative': {'GC=F': 0.40, 'SI=F': 0.30, 'CL=F': 0.15, 'HG=F': 0.15},
        'moderate':     {'GC=F': 0.30, 'CL=F': 0.25, 'SI=F': 0.20, 'HG=F': 0.15, 'NG=F': 0.10},
        'aggressive':   {'CL=F': 0.25, 'GC=F': 0.20, 'NG=F': 0.15, 'SI=F': 0.15, 'HG=F': 0.15, 'PL=F': 0.10},
    },
}

# Static expected return estimates by risk level
_STATIC_RETURNS = {
    'conservative': {'return': 0.06, 'volatility': 0.08},
    'moderate':     {'return': 0.10, 'volatility': 0.15},
    'aggressive':   {'return': 0.15, 'volatility': 0.22},
}


class PortfolioOptimizer:
    """Portfolio optimization using PyPortfolioOpt with Alpha Vantage primary data."""

    def __init__(self, av_collector=None):
        self.av_collector = av_collector

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
        elif market == 'crypto':
            pool = SYMBOL_POOLS_CRYPTO
        elif market == 'commodity':
            pool = SYMBOL_POOLS_COMMODITY
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

        # Filter to symbols with available data using Alpha Vantage (fast)
        valid_symbols = self._validate_symbols(symbols)

        if len(valid_symbols) < 2:
            logger.warning("Insufficient live data, using static fallback allocation")
            return self._static_fallback(amount, risk_level, market)

        # Optimize
        try:
            result = self.optimize(valid_symbols, method)
        except Exception as e:
            logger.warning(f"Optimization failed ({e}), using static fallback")
            return self._static_fallback(amount, risk_level, market)

        weights = result['weights']

        # Compute allocations with latest prices
        allocations = {}
        for symbol, weight in weights.items():
            sym_amount = amount * weight
            latest_price = self._get_latest_price(symbol)
            shares_approx = (sym_amount / latest_price) if latest_price else 0

            allocations[symbol] = {
                'weight': weight,
                'amount': round(sym_amount, 2),
                'shares_approx': round(shares_approx, 2),
            }

        # Risk metrics on the recommended allocation
        try:
            risk_metrics = self.risk_analysis(
                list(weights.keys()),
                list(weights.values())
            )
        except Exception as e:
            logger.warning(f"Risk analysis failed: {e}")
            risk_metrics = {}

        # Expected return ranges
        annual_ret = result['expected_annual_return']
        annual_vol = result['annual_volatility']

        # Determine currency from the market
        currency = 'INR' if market == 'india' else 'USD'
        currency_symbol = '\u20b9' if market == 'india' else '$'

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

    def _validate_symbols(self, symbols: List[str]) -> List[str]:
        """Validate which symbols have available data. Alpha Vantage first, yfinance fallback."""
        valid = []

        # Try Alpha Vantage first (fast, reliable on Railway)
        if self.av_collector:
            for s in symbols:
                try:
                    quote = self.av_collector.get_quote(s)
                    if quote and quote.get('price', 0) > 0:
                        valid.append(s)
                        continue
                except Exception:
                    pass
            if len(valid) >= 2:
                logger.info(f"Validated {len(valid)}/{len(symbols)} symbols via Alpha Vantage")
                return valid

        # Fallback: yfinance batch download (single request, faster than individual)
        try:
            import yfinance as yf
            yf_symbols = [_to_yf_symbol(s) for s in symbols]
            data = yf.download(yf_symbols, period="5d", progress=False, threads=True)
            if not data.empty:
                # Map back to original symbols
                av_to_yf = {s: _to_yf_symbol(s) for s in symbols}
                for s in symbols:
                    yf_s = av_to_yf[s]
                    try:
                        if 'Close' in data.columns:
                            col = data['Close'] if len(yf_symbols) == 1 else data['Close'][yf_s]
                        else:
                            col = data[yf_s] if yf_s in data.columns else None
                        if col is not None and not col.dropna().empty:
                            if s not in valid:
                                valid.append(s)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"yfinance batch validation failed: {e}")

        logger.info(f"Validated {len(valid)}/{len(symbols)} symbols total")
        return valid

    def _get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol. Alpha Vantage first."""
        if self.av_collector:
            try:
                quote = self.av_collector.get_quote(symbol)
                if quote and quote.get('price', 0) > 0:
                    return quote['price']
            except Exception:
                pass

        try:
            import yfinance as yf
            ticker = yf.Ticker(_to_yf_symbol(symbol))
            hist = ticker.history(period="5d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except Exception:
            pass

        return None

    def _get_price_data(self, symbols: List[str], period_days: int = 504) -> pd.DataFrame:
        """Fetch adjusted close prices. Alpha Vantage primary, yfinance fallback."""
        prices = {}

        # Try Alpha Vantage first for each symbol
        if self.av_collector:
            for symbol in symbols:
                try:
                    df = self.av_collector.get_daily_history(symbol, days=period_days)
                    if df is not None and not df.empty and 'close' in df.columns:
                        prices[symbol] = df['close']
                except Exception as e:
                    logger.warning(f"Alpha Vantage history failed for {symbol}: {e}")

        # Fallback to yfinance for any missing symbols
        missing = [s for s in symbols if s not in prices]
        if missing:
            for symbol in missing:
                try:
                    import yfinance as yf
                    ticker = yf.Ticker(_to_yf_symbol(symbol))
                    hist = ticker.history(period=f"{period_days}d")
                    if not hist.empty:
                        prices[symbol] = hist['Close']
                except Exception as e:
                    logger.warning(f"yfinance history failed for {symbol}: {e}")

        if not prices:
            return pd.DataFrame()

        return pd.DataFrame(prices).dropna()

    def _static_fallback(
        self,
        amount: float,
        risk_level: str,
        market: Optional[str],
    ) -> Dict[str, Any]:
        """Return a static allocation when live data is unavailable."""
        market_key = market if market in ('us', 'india') else 'global'
        weights = _STATIC_ALLOCATIONS[market_key][risk_level]
        estimates = _STATIC_RETURNS[risk_level]

        allocations = {}
        for symbol, weight in weights.items():
            sym_amount = amount * weight
            allocations[symbol] = {
                'weight': weight,
                'amount': round(sym_amount, 2),
                'shares_approx': 0,  # Can't compute without live price
            }

        currency = 'INR' if market == 'india' else 'USD'
        currency_symbol = '\u20b9' if market == 'india' else '$'
        annual_ret = estimates['return']
        annual_vol = estimates['volatility']

        return {
            'allocations': allocations,
            'risk_level': risk_level,
            'market': market_key,
            'currency': currency,
            'currency_symbol': currency_symbol,
            'expected_return_range': {
                'low': annual_ret - annual_vol,
                'mid': annual_ret,
                'high': annual_ret + annual_vol,
            },
            'risk_metrics': {
                'annual_volatility': annual_vol,
                'sharpe_ratio': annual_ret / annual_vol if annual_vol else 0,
            },
            'method': 'static_fallback',
        }

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
            # Try Alpha Vantage for SPY data
            if self.av_collector:
                df = self.av_collector.get_daily_history('SPY', days=len(port_returns) + 30)
                if df is not None and not df.empty and 'close' in df.columns:
                    spy_returns = df['close'].pct_change().dropna()
                    aligned = pd.DataFrame({
                        'port': port_returns,
                        'spy': spy_returns
                    }).dropna()
                    if len(aligned) >= 30:
                        cov = aligned['port'].cov(aligned['spy'])
                        var = aligned['spy'].var()
                        return float(cov / var) if var > 0 else 1.0

            # Fallback to yfinance
            import yfinance as yf
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
