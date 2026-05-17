#!/usr/bin/env python3
"""
Configuration File for AI Stock Prediction System
================================================

This file contains all configurable parameters for the AI Stock Prediction System.
Modify these settings to customize the behavior of the system.

Author: AI Assistant
Date: 2024
"""

# =============================================================================
# DATA COLLECTION CONFIGURATION
# =============================================================================

# Default stock symbol to analyze
DEFAULT_SYMBOL = 'AAPL'

# Default date range for data collection
DEFAULT_START_DATE = '2020-01-01'
DEFAULT_END_DATE = None  # None means current date

# Data collection settings
DATA_CONFIG = {
    'symbol': DEFAULT_SYMBOL,
    'start_date': DEFAULT_START_DATE,
    'end_date': DEFAULT_END_DATE,
    'sequence_length': 60,  # Number of time steps for LSTM
    'test_size': 0.2,       # Proportion of data for testing
    'target_column': 'Close',  # Target variable to predict
    'validation_split': 0.2,   # Proportion for validation during training
}

# Technical indicators to include
TECHNICAL_INDICATORS = {
    'moving_averages': {
        'SMA': [20, 50],
        'EMA': [12, 26]
    },
    'momentum': {
        'RSI': 14,
        'MACD': True,
        'Stochastic': True
    },
    'volatility': {
        'Bollinger_Bands': True,
        'ATR': True
    },
    'volume': {
        'Volume_SMA': True,
        'OBV': True
    }
}

# Feature engineering settings
FEATURE_ENGINEERING = {
    'lag_features': [1, 2, 3, 5, 10],  # Days to lag
    'rolling_windows': [5, 10, 20],     # Rolling statistics windows
    'time_features': True,              # Include day/week/month features
    'price_changes': True,              # Include price change features
    'volatility_features': True         # Include volatility features
}

# =============================================================================
# LSTM MODEL CONFIGURATION
# =============================================================================

# Default model architecture
MODEL_CONFIG = {
    'lstm_units': [128, 64],      # LSTM layer sizes
    'dropout_rate': 0.2,          # Dropout for regularization
    'learning_rate': 0.001,       # Adam optimizer learning rate
    'batch_size': 32,             # Batch size for training
    'epochs': 100,                # Number of training epochs
    'early_stopping_patience': 15,  # Early stopping patience
    'reduce_lr_patience': 10,     # Learning rate reduction patience
    'min_lr': 1e-7,              # Minimum learning rate
    'bidirectional': True,        # Use bidirectional LSTM
    'batch_normalization': True,  # Use batch normalization
}

# Model variants for ensemble
ENSEMBLE_MODELS = [
    {
        'name': 'small',
        'lstm_units': [64, 32],
        'dropout_rate': 0.1,
        'learning_rate': 0.001
    },
    {
        'name': 'medium',
        'lstm_units': [128, 64],
        'dropout_rate': 0.2,
        'learning_rate': 0.001
    },
    {
        'name': 'large',
        'lstm_units': [256, 128, 64],
        'dropout_rate': 0.3,
        'learning_rate': 0.0005
    },
    {
        'name': 'deep',
        'lstm_units': [512, 256, 128, 64],
        'dropout_rate': 0.4,
        'learning_rate': 0.0001
    }
]

# =============================================================================
# SENSITIVITY ANALYSIS CONFIGURATION
# =============================================================================

# Sensitivity analysis methods
SENSITIVITY_CONFIG = {
    'methods': ['perturbation', 'correlation', 'gradient'],
    'perturbation_levels': [0.1, 0.2, 0.5, 1.0],
    'sample_size': 100,  # Number of samples for sensitivity analysis
    'top_features': 10,  # Number of top features to display
}

# Market condition scenarios
MARKET_SCENARIOS = {
    'High_Volatility': {
        0: {'type': 'noise', 'value': 0.5},  # Price volatility
        4: {'type': 'multiply', 'value': 1.5}  # Volume increase
    },
    'Low_Volatility': {
        0: {'type': 'noise', 'value': 0.1},  # Price stability
        4: {'type': 'multiply', 'value': 0.5}  # Volume decrease
    },
    'Trend_Up': {
        0: {'type': 'add', 'value': 0.1},  # Price increase
        1: {'type': 'add', 'value': 0.1},  # High increase
        2: {'type': 'add', 'value': 0.1}   # Low increase
    },
    'Trend_Down': {
        0: {'type': 'add', 'value': -0.1},  # Price decrease
        1: {'type': 'add', 'value': -0.1},  # High decrease
        2: {'type': 'add', 'value': -0.1}   # Low decrease
    },
    'Market_Crash': {
        0: {'type': 'add', 'value': -0.3},  # Sharp price drop
        4: {'type': 'multiply', 'value': 2.0}  # High volume
    },
    'Market_Rally': {
        0: {'type': 'add', 'value': 0.3},   # Sharp price increase
        4: {'type': 'multiply', 'value': 2.0}  # High volume
    }
}

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

# Performance metrics to calculate
EVALUATION_METRICS = [
    'MSE', 'RMSE', 'MAE', 'R2', 'MAPE', 'Directional_Accuracy'
]

# Prediction settings
PREDICTION_CONFIG = {
    'days_ahead': 5,              # Default prediction horizon
    'confidence_interval': 0.95,  # Confidence interval for predictions
    'ensemble_size': 5,           # Number of models for ensemble
}

# =============================================================================
# VISUALIZATION CONFIGURATION
# =============================================================================

# Plot settings
PLOT_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'color_palette': 'viridis',
    'save_format': 'png',
    'show_plots': True,
}

# Chart types to generate
CHARTS_TO_GENERATE = [
    'training_history',
    'predictions_vs_actual',
    'feature_importance',
    'sensitivity_analysis',
    'market_conditions',
    'prediction_confidence'
]

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

# Output directories
OUTPUT_CONFIG = {
    'models_dir': 'models',
    'reports_dir': 'reports',
    'plots_dir': 'reports/plots',
    'logs_dir': 'logs',
    'data_dir': 'data'
}

# Report settings
REPORT_CONFIG = {
    'generate_html': True,
    'generate_pdf': False,
    'include_plots': True,
    'include_metrics': True,
    'include_predictions': True,
    'include_sensitivity': True
}

# =============================================================================
# STOCK-SPECIFIC CONFIGURATIONS
# =============================================================================

# Predefined configurations for different stocks
STOCK_CONFIGS = {
    'AAPL': {
        'name': 'Apple Inc.',
        'sector': 'Technology',
        'volatility': 'medium',
        'recommended_sequence_length': 60,
        'recommended_epochs': 100
    },
    'MSFT': {
        'name': 'Microsoft Corporation',
        'sector': 'Technology',
        'volatility': 'medium',
        'recommended_sequence_length': 60,
        'recommended_epochs': 100
    },
    'GOOGL': {
        'name': 'Alphabet Inc.',
        'sector': 'Technology',
        'volatility': 'medium',
        'recommended_sequence_length': 60,
        'recommended_epochs': 100
    },
    'TSLA': {
        'name': 'Tesla Inc.',
        'sector': 'Automotive',
        'volatility': 'high',
        'recommended_sequence_length': 90,
        'recommended_epochs': 150
    },
    'AMZN': {
        'name': 'Amazon.com Inc.',
        'sector': 'Consumer Discretionary',
        'volatility': 'medium',
        'recommended_sequence_length': 60,
        'recommended_epochs': 100
    },
    'NVDA': {
        'name': 'NVIDIA Corporation',
        'sector': 'Technology',
        'volatility': 'high',
        'recommended_sequence_length': 90,
        'recommended_epochs': 150
    },
    'JPM': {
        'name': 'JPMorgan Chase & Co.',
        'sector': 'Financial',
        'volatility': 'medium',
        'recommended_sequence_length': 60,
        'recommended_epochs': 100
    },
    'JNJ': {
        'name': 'Johnson & Johnson',
        'sector': 'Healthcare',
        'volatility': 'low',
        'recommended_sequence_length': 45,
        'recommended_epochs': 80
    }
}

# =============================================================================
# DAY TRADING CONFIGURATION
# =============================================================================

DAY_TRADING_CONFIG = {
    'interval': '5min',             # Intraday bar interval
    'months_history': 3,            # Months of intraday data to fetch
    'target_horizon_bars': 12,      # Forward look-ahead in bars (12 x 5min = 60 min)
    'min_training_rows': 500,       # Minimum intraday bars for training
    'rate_limit_delay': 12.5,       # Seconds between Alpha Vantage API calls
}

# Liquid symbols suitable for day trading (US only -- Alpha Vantage intraday
# is limited to US equities; India BSE symbols use daily models via train_models.py)
DAY_TRADING_SYMBOLS = [
    # High-liquidity US tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD',
    # High-volume ETFs
    'SPY', 'QQQ', 'IWM',
    # Financials
    'JPM', 'BAC',
    # Energy
    'XOM',
]

DAY_TRADING_QUICK_SYMBOLS = ['AAPL', 'MSFT', 'SPY', 'TSLA', 'NVDA']

# =============================================================================
# ADVANCED CONFIGURATION
# =============================================================================

# Advanced model settings
ADVANCED_CONFIG = {
    'use_attention': False,        # Use attention mechanism
    'use_residual_connections': True,  # Use residual connections
    'use_layer_norm': False,      # Use layer normalization
    'gradient_clipping': 1.0,     # Gradient clipping value
    'weight_decay': 0.0001,       # L2 regularization
    'label_smoothing': 0.0,       # Label smoothing
}

# Data preprocessing settings
PREPROCESSING_CONFIG = {
    'normalization': 'minmax',     # 'minmax', 'standard', 'robust'
    'handle_missing': 'interpolate',  # 'drop', 'interpolate', 'forward_fill'
    'outlier_detection': True,     # Detect and handle outliers
    'feature_selection': False,    # Use feature selection
    'dimensionality_reduction': False,  # Use PCA or other methods
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',              # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/stock_prediction.log',
    'console': True,
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_stock_config(symbol):
    """
    Get configuration for a specific stock.
    
    Args:
        symbol (str): Stock symbol
        
    Returns:
        dict: Stock-specific configuration
    """
    return STOCK_CONFIGS.get(symbol.upper(), {
        'name': f'{symbol}',
        'sector': 'Unknown',
        'volatility': 'medium',
        'recommended_sequence_length': 60,
        'recommended_epochs': 100
    })

def get_model_config_for_stock(symbol):
    """
    Get optimized model configuration for a specific stock.
    
    Args:
        symbol (str): Stock symbol
        
    Returns:
        dict: Optimized model configuration
    """
    stock_config = get_stock_config(symbol)
    
    # Adjust model parameters based on stock volatility
    volatility = stock_config['volatility']
    
    if volatility == 'high':
        return {
            'lstm_units': [256, 128, 64],
            'dropout_rate': 0.3,
            'learning_rate': 0.0005,
            'epochs': stock_config['recommended_epochs'],
            'sequence_length': stock_config['recommended_sequence_length']
        }
    elif volatility == 'low':
        return {
            'lstm_units': [64, 32],
            'dropout_rate': 0.1,
            'learning_rate': 0.001,
            'epochs': stock_config['recommended_epochs'],
            'sequence_length': stock_config['recommended_sequence_length']
        }
    else:  # medium volatility
        return {
            'lstm_units': [128, 64],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'epochs': stock_config['recommended_epochs'],
            'sequence_length': stock_config['recommended_sequence_length']
        }

def validate_config():
    """
    Validate the configuration settings.
    
    Returns:
        bool: True if configuration is valid
    """
    errors = []
    
    # Validate data configuration
    if DATA_CONFIG['sequence_length'] <= 0:
        errors.append("sequence_length must be positive")
    
    if not (0 < DATA_CONFIG['test_size'] < 1):
        errors.append("test_size must be between 0 and 1")
    
    # Validate model configuration
    if MODEL_CONFIG['dropout_rate'] < 0 or MODEL_CONFIG['dropout_rate'] > 1:
        errors.append("dropout_rate must be between 0 and 1")
    
    if MODEL_CONFIG['learning_rate'] <= 0:
        errors.append("learning_rate must be positive")
    
    # Validate sensitivity configuration
    if SENSITIVITY_CONFIG['sample_size'] <= 0:
        errors.append("sample_size must be positive")
    
    if errors:
        print("Configuration errors found:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True

# Validate configuration on import
if __name__ == "__main__":
    if validate_config():
        print("✅ Configuration is valid!")
    else:
        print("❌ Configuration has errors!")
else:
    # Only validate when imported as module
    validate_config()
