"""
XGBoost model with walk-forward validation for stock prediction.
"""

import os
import logging
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


class XGBoostPredictor:
    """XGBoost model with walk-forward validation for stock prediction."""

    def __init__(self, models_dir: str = 'models'):
        self.models_dir = models_dir
        self.models = {}  # {symbol: XGBClassifier}
        os.makedirs(models_dir, exist_ok=True)

    def train(self, symbol: str, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Walk-forward validation: 252-day train window, 21-day test window, rolling.
        Target: binary -- 1 if 21-day forward return > 0, else 0.
        """
        target = self._create_target(features_df)
        feature_cols = [c for c in features_df.columns if c not in [
            'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits'
        ]]

        X = features_df[feature_cols].values
        y = target.values

        # Remove rows where target is NaN
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]

        train_window = 252
        test_window = 21

        all_preds = []
        all_actuals = []

        # Walk-forward validation
        for start in range(0, len(X) - train_window - test_window, test_window):
            train_end = start + train_window
            test_end = min(train_end + test_window, len(X))

            X_train = X[start:train_end]
            y_train = y[start:train_end]
            X_test = X[train_end:test_end]
            y_test = y[train_end:test_end]

            model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42,
            )
            model.fit(X_train, y_train, verbose=False)

            preds = model.predict(X_test)
            all_preds.extend(preds)
            all_actuals.extend(y_test)

        # Train final model on all data
        final_model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
        )
        final_model.fit(X, y, verbose=False)
        self.models[symbol] = final_model

        # Compute metrics
        all_preds = np.array(all_preds)
        all_actuals = np.array(all_actuals)

        # Feature importance
        importance = dict(zip(feature_cols, final_model.feature_importances_))
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])

        self.save_model(symbol)

        return {
            'accuracy': float(accuracy_score(all_actuals, all_preds)),
            'precision': float(precision_score(all_actuals, all_preds, zero_division=0)),
            'recall': float(recall_score(all_actuals, all_preds, zero_division=0)),
            'f1': float(f1_score(all_actuals, all_preds, zero_division=0)),
            'feature_importance': sorted_importance,
        }

    def predict(self, symbol: str, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Load model, predict on latest features.
        Returns prediction dict with direction, probability, confidence.
        """
        # Try to load model if not in memory
        if symbol not in self.models:
            if not self.load_model(symbol):
                # Auto-train if no saved model
                logger.info(f"No saved model for {symbol}, training...")
                metrics = self.train(symbol, features_df)
                logger.info(f"Trained {symbol}: accuracy={metrics['accuracy']:.2%}")

        model = self.models.get(symbol)
        if model is None:
            raise ValueError(f"No model available for {symbol}")

        feature_cols = [c for c in features_df.columns if c not in [
            'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits'
        ]]

        X_latest = features_df[feature_cols].iloc[[-1]].values
        proba = model.predict_proba(X_latest)[0]
        pred_class = model.predict(X_latest)[0]

        # Probability of positive return
        prob_positive = float(proba[1]) if len(proba) > 1 else float(proba[0])

        # Feature importance
        importance = dict(zip(feature_cols, model.feature_importances_))
        top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])

        # Confidence based on distance from 0.5
        confidence = abs(prob_positive - 0.5) * 2

        # Expected return estimate (calibrated)
        returns = features_df['close'].pct_change(21).dropna()
        if len(returns) > 0:
            avg_pos_return = returns[returns > 0].mean() if (returns > 0).any() else 0.02
            avg_neg_return = returns[returns <= 0].mean() if (returns <= 0).any() else -0.02
            expected_return = prob_positive * avg_pos_return + (1 - prob_positive) * avg_neg_return
        else:
            expected_return = 0.0

        return {
            'symbol': symbol,
            'direction': 'bullish' if pred_class == 1 else 'bearish',
            'probability': prob_positive,
            'expected_return': float(expected_return),
            'confidence': confidence,
            'feature_importance': top_features,
            'horizon_days': 21,
        }

    def load_model(self, symbol: str) -> bool:
        """Load a saved model from disk."""
        path = os.path.join(self.models_dir, f'{symbol}_xgboost.joblib')
        if os.path.exists(path):
            try:
                self.models[symbol] = joblib.load(path)
                logger.info(f"Loaded XGBoost model for {symbol}")
                return True
            except Exception as e:
                logger.error(f"Failed to load model for {symbol}: {e}")
        return False

    def save_model(self, symbol: str) -> bool:
        """Save a model to disk."""
        if symbol not in self.models:
            return False
        try:
            path = os.path.join(self.models_dir, f'{symbol}_xgboost.joblib')
            joblib.dump(self.models[symbol], path)
            logger.info(f"Saved XGBoost model for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model for {symbol}: {e}")
            return False

    def _create_target(self, df: pd.DataFrame, horizon: int = 21) -> pd.Series:
        """Create binary target: 1 if 21-day forward return > 0, else 0."""
        forward_return = df['close'].pct_change(horizon).shift(-horizon)
        return (forward_return > 0).astype(float)
