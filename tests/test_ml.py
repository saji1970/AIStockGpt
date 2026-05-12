"""Smoke tests for ML engine modules."""

import pytest


def test_feature_pipeline_import():
    from backend.ml.feature_pipeline import FeaturePipeline
    fp = FeaturePipeline()
    assert fp is not None


def test_xgboost_import():
    from backend.ml.xgboost_model import XGBoostPredictor
    predictor = XGBoostPredictor()
    assert predictor is not None


def test_monte_carlo_import():
    from backend.ml.monte_carlo import MonteCarloSimulator
    mc = MonteCarloSimulator()
    assert mc is not None


def test_portfolio_optimizer_import():
    from backend.ml.portfolio_optimizer import PortfolioOptimizer
    opt = PortfolioOptimizer()
    assert opt is not None


def test_sentiment_import():
    from backend.ml.sentiment import SentimentAnalyzer
    sa = SentimentAnalyzer()
    assert sa is not None


def test_fred_collector_import():
    from backend.data.fred_collector import FREDCollector
    fc = FREDCollector()
    assert fc is not None


def test_finnhub_collector_import():
    from backend.data.finnhub_collector import FinnhubCollector
    fc = FinnhubCollector()
    assert fc is not None


def test_scheduler_import():
    from backend.data.scheduler import DataScheduler
    # Cannot instantiate without db_manager, just verify import
    assert DataScheduler is not None


@pytest.mark.slow
def test_monte_carlo_basic():
    """Test Monte Carlo simulation with a small number of simulations."""
    from backend.ml.monte_carlo import MonteCarloSimulator
    mc = MonteCarloSimulator()
    result = mc.simulate(['AAPL'], [1.0], 1000, 6, n_simulations=100)
    assert 'median_value' in result
    assert result['probability_positive'] >= 0
    assert result['initial_amount'] == 1000
    assert result['months'] == 6


@pytest.mark.slow
def test_portfolio_optimizer_basic():
    """Test portfolio optimization with real data."""
    from backend.ml.portfolio_optimizer import PortfolioOptimizer
    opt = PortfolioOptimizer()
    result = opt.optimize(['AAPL', 'MSFT', 'BND'], 'max_sharpe')
    assert abs(sum(result['weights'].values()) - 1.0) < 0.01
    assert 'expected_annual_return' in result
    assert 'sharpe_ratio' in result
