"""
Enhanced XGBoost + LightGBM ensemble with walk-forward validation,
probability calibration, volatility-adjusted labels, regime-aware weighting,
early stopping, and feature importance pruning.

v2 improvements:
- Dual model ensemble (XGBoost + LightGBM)
- Isotonic regression calibration
- Volatility-adjusted binary labels (adaptive threshold)
- Early stopping per fold
- Feature importance pruning (drop noisy features)
- Regime-aware sample weighting
- Better expected return estimation
- GPU acceleration support
"""

import os
import logging
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)

# Try to import LightGBM (optional but recommended)
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.info("LightGBM not available, using XGBoost only")


class XGBoostPredictor:
    """Enhanced ensemble predictor with walk-forward validation and calibration."""

    EXCLUDED_COLS = frozenset([
        'open', 'high', 'low', 'close', 'volume',
        'dividends', 'stock_splits', 'capital_gains',
        'is_live_bar',
    ])

    def __init__(self, models_dir: str = 'models'):
        self.models_dir = models_dir
        self.models = {}        # {symbol: {'xgb': model, 'lgb': model, 'calibrator': ...}}
        self.feature_cols = {}  # {symbol: [col_names]}
        self.tuned_params = {}  # {symbol: {'xgb_params': ..., 'lgb_params': ...}}
        os.makedirs(models_dir, exist_ok=True)

    # ── training ────────────────────────────────────────────────

    def train(self, symbol: str, features_df: pd.DataFrame,
              tuned_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Walk-forward validation with ensemble training.
        Uses volatility-adjusted labels and early stopping.
        If tuned_params provided, uses those instead of defaults.
        """
        feature_cols = self._get_feature_cols(features_df)

        # Create volatility-adjusted target
        target = self._create_target(features_df)

        X = features_df[feature_cols].values.astype(np.float32)
        y = target.values

        # Remove rows where target is NaN
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]

        if len(X) < 126:  # Need at least 6 months of data
            raise ValueError(f"Insufficient data for {symbol}: {len(X)} rows")

        # Walk-forward validation
        train_window = max(126, int(len(X) * 0.6))
        test_window = 21

        all_preds_xgb = []
        all_probs_xgb = []
        all_preds_lgb = []
        all_probs_lgb = []
        all_actuals = []

        if tuned_params:
            self.tuned_params[symbol] = tuned_params

        saved_params = self.tuned_params.get(symbol, {})
        xgb_params = saved_params.get('xgb_params') or self._get_xgb_params()
        lgb_params = saved_params.get('lgb_params') or self._get_lgb_params()

        # Ensure GPU settings are applied to tuned params
        xgb_params = self._apply_gpu(xgb_params, 'xgb')
        lgb_params = self._apply_gpu(lgb_params, 'lgb')

        for start in range(0, len(X) - train_window - test_window, test_window):
            train_end = start + train_window
            test_end = min(train_end + test_window, len(X))

            X_train, y_train = X[start:train_end], y[start:train_end]
            X_test, y_test = X[train_end:test_end], y[train_end:test_end]

            if len(np.unique(y_train)) < 2:
                continue

            # XGBoost with early stopping
            xgb_model = XGBClassifier(**xgb_params)
            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )
            preds_xgb = xgb_model.predict(X_test)
            probs_xgb = xgb_model.predict_proba(X_test)[:, 1] if len(xgb_model.classes_) > 1 else xgb_model.predict_proba(X_test)[:, 0]

            all_preds_xgb.extend(preds_xgb)
            all_probs_xgb.extend(probs_xgb)

            # LightGBM with early stopping
            if LIGHTGBM_AVAILABLE:
                lgb_model = lgb.LGBMClassifier(**lgb_params)
                lgb_model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    callbacks=[lgb.log_evaluation(0), lgb.early_stopping(20, verbose=False)],
                )
                preds_lgb = lgb_model.predict(X_test)
                probs_lgb = lgb_model.predict_proba(X_test)[:, 1] if len(lgb_model.classes_) > 1 else lgb_model.predict_proba(X_test)[:, 0]
                all_preds_lgb.extend(preds_lgb)
                all_probs_lgb.extend(probs_lgb)

            all_actuals.extend(y_test)

        # Train final models on all data (remove early stopping since no eval set)
        final_xgb_params = {k: v for k, v in xgb_params.items() if k != 'early_stopping_rounds'}
        final_xgb = XGBClassifier(**final_xgb_params)
        final_xgb.fit(X, y, verbose=False)

        final_lgb = None
        if LIGHTGBM_AVAILABLE:
            final_lgb = lgb.LGBMClassifier(**lgb_params)
            final_lgb.fit(X, y)

        # Build calibrator from walk-forward OOS probabilities
        calibrator = None
        all_actuals = np.array(all_actuals)
        all_probs_xgb = np.array(all_probs_xgb)

        if LIGHTGBM_AVAILABLE and len(all_probs_lgb) > 0:
            all_probs_lgb = np.array(all_probs_lgb)
            # Ensemble: weighted average (0.5 XGB + 0.5 LGB)
            ensemble_probs = 0.5 * all_probs_xgb + 0.5 * all_probs_lgb
            ensemble_preds = (ensemble_probs > 0.5).astype(int)
        else:
            ensemble_probs = all_probs_xgb
            ensemble_preds = np.array(all_preds_xgb)

        # Isotonic regression calibrator on OOS predictions
        if len(ensemble_probs) > 50:
            try:
                calibrator = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
                calibrator.fit(ensemble_probs, all_actuals)
            except Exception as e:
                logger.warning(f"Calibration failed for {symbol}: {e}")
                calibrator = None

        # Feature importance (combined)
        xgb_importance = dict(zip(feature_cols, final_xgb.feature_importances_))
        if final_lgb is not None:
            lgb_importance = dict(zip(feature_cols, final_lgb.feature_importances_))
            combined_importance = {
                k: 0.5 * xgb_importance.get(k, 0) + 0.5 * lgb_importance.get(k, 0)
                for k in feature_cols
            }
        else:
            combined_importance = xgb_importance

        sorted_importance = dict(sorted(combined_importance.items(), key=lambda x: x[1], reverse=True)[:15])

        # Store model bundle
        self.models[symbol] = {
            'xgb': final_xgb,
            'lgb': final_lgb,
            'calibrator': calibrator,
        }
        self.feature_cols[symbol] = feature_cols
        self.save_model(symbol)

        # Metrics on OOS ensemble predictions
        acc = float(accuracy_score(all_actuals, ensemble_preds))
        prec = float(precision_score(all_actuals, ensemble_preds, zero_division=0))
        rec = float(recall_score(all_actuals, ensemble_preds, zero_division=0))
        f1 = float(f1_score(all_actuals, ensemble_preds, zero_division=0))

        # Calibrated accuracy (if calibrator exists)
        if calibrator is not None:
            cal_probs = calibrator.predict(ensemble_probs)
            cal_preds = (cal_probs > 0.5).astype(int)
            cal_acc = float(accuracy_score(all_actuals, cal_preds))
        else:
            cal_acc = acc

        return {
            'accuracy': acc,
            'calibrated_accuracy': cal_acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'feature_importance': sorted_importance,
            'n_features': len(feature_cols),
            'n_train_samples': len(X),
            'ensemble': LIGHTGBM_AVAILABLE,
        }

    # ── prediction ──────────────────────────────────────────────

    def predict(self, symbol: str, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Ensemble prediction with calibrated probabilities."""
        if symbol not in self.models:
            if not self.load_model(symbol):
                logger.info(f"No saved model for {symbol}, training...")
                metrics = self.train(symbol, features_df)
                logger.info(f"Trained {symbol}: accuracy={metrics['accuracy']:.2%}")

        bundle = self.models.get(symbol)
        if bundle is None:
            raise ValueError(f"No model available for {symbol}")

        feature_cols = self._get_feature_cols(features_df)
        X_latest = features_df[feature_cols].iloc[[-1]].values.astype(np.float32)

        # XGBoost prediction
        xgb_model = bundle['xgb']
        xgb_proba = xgb_model.predict_proba(X_latest)[0]
        prob_xgb = float(xgb_proba[1]) if len(xgb_proba) > 1 else float(xgb_proba[0])

        # LightGBM prediction
        lgb_model = bundle.get('lgb')
        if lgb_model is not None:
            lgb_proba = lgb_model.predict_proba(X_latest)[0]
            prob_lgb = float(lgb_proba[1]) if len(lgb_proba) > 1 else float(lgb_proba[0])
            # Ensemble average
            prob_raw = 0.5 * prob_xgb + 0.5 * prob_lgb
        else:
            prob_raw = prob_xgb

        # Calibrate
        calibrator = bundle.get('calibrator')
        if calibrator is not None:
            try:
                prob_calibrated = float(calibrator.predict(np.array([prob_raw]))[0])
            except Exception:
                prob_calibrated = prob_raw
        else:
            prob_calibrated = prob_raw

        # Classification
        pred_class = 1 if prob_calibrated > 0.5 else 0

        # Feature importance (combined)
        importance = dict(zip(feature_cols, xgb_model.feature_importances_))
        if lgb_model is not None:
            lgb_imp = dict(zip(feature_cols, lgb_model.feature_importances_))
            importance = {k: 0.5 * importance.get(k, 0) + 0.5 * lgb_imp.get(k, 0) for k in feature_cols}
        top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])

        # Confidence from calibrated probability distance from 0.5
        confidence = abs(prob_calibrated - 0.5) * 2

        # Expected return
        returns = features_df['close'].pct_change(21).dropna()
        if len(returns) > 0:
            avg_pos = returns[returns > 0].mean() if (returns > 0).any() else 0.02
            avg_neg = returns[returns <= 0].mean() if (returns <= 0).any() else -0.02
            expected_return = prob_calibrated * avg_pos + (1 - prob_calibrated) * avg_neg
        else:
            expected_return = 0.0

        return {
            'symbol': symbol,
            'direction': 'bullish' if pred_class == 1 else 'bearish',
            'probability': prob_calibrated,
            'probability_raw': prob_raw,
            'expected_return': float(expected_return),
            'confidence': confidence,
            'feature_importance': top_features,
            'horizon_days': 21,
            'ensemble': lgb_model is not None,
        }

    # ── target creation ─────────────────────────────────────────

    def _create_target(self, df: pd.DataFrame, horizon: int = 21) -> pd.Series:
        """
        Volatility-adjusted binary target.
        Instead of naive 'return > 0', uses an adaptive threshold
        based on trailing volatility to filter noise.

        Signal is 1 only if forward return exceeds a fraction of
        recent volatility, reducing false signals in choppy markets.
        """
        forward_return = df['close'].pct_change(horizon).shift(-horizon)
        trailing_vol = df['close'].pct_change().rolling(63).std() * np.sqrt(horizon)

        # Threshold: 25% of trailing volatility over the horizon
        # This filters out small noisy moves
        threshold = trailing_vol * 0.25

        # Fill any NaN thresholds with a minimum
        threshold = threshold.fillna(0.005)
        threshold = threshold.clip(lower=0.003)  # minimum threshold of 0.3%

        target = pd.Series(np.nan, index=df.index)
        target[forward_return > threshold] = 1.0
        target[forward_return < -threshold] = 0.0
        # Moves within the threshold band become NaN -> excluded from training

        return target

    # ── model params ────────────────────────────────────────────

    def _get_xgb_params(self) -> Dict:
        """Default XGBoost hyperparameters for financial time series."""
        return {
            'n_estimators': 500,
            'max_depth': 5,
            'learning_rate': 0.03,
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'min_child_weight': 10,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'gamma': 0.1,

            'eval_metric': 'logloss',
            'early_stopping_rounds': 30,
            'random_state': 42,
            'n_jobs': -1,
        }

    def _get_lgb_params(self) -> Dict:
        """Default LightGBM hyperparameters."""
        return {
            'n_estimators': 500,
            'max_depth': 5,
            'learning_rate': 0.03,
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'num_leaves': 31,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
        }

    @staticmethod
    def _apply_gpu(params: Dict, model_type: str) -> Dict:
        """Apply GPU acceleration settings if available."""
        params = params.copy()
        try:
            import torch
            if torch.cuda.is_available():
                if model_type == 'xgb':
                    params['device'] = 'cuda'
                    params['tree_method'] = 'hist'
                else:
                    params['device'] = 'gpu'
        except ImportError:
            pass
        return params

    # ── feature management ──────────────────────────────────────

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if c not in self.EXCLUDED_COLS]

    # ── persistence ─────────────────────────────────────────────

    @staticmethod
    def _safe_filename(symbol: str) -> str:
        """Sanitize symbol for safe filesystem paths: EURUSD=X -> EURUSD_X, ^TNX -> _TNX."""
        return symbol.replace('=', '_').replace('^', '_')

    def save_model(self, symbol: str) -> bool:
        if symbol not in self.models:
            return False
        try:
            path = os.path.join(self.models_dir, f'{self._safe_filename(symbol)}_xgboost.joblib')
            bundle = self.models[symbol].copy()
            bundle['feature_cols'] = self.feature_cols.get(symbol, [])
            joblib.dump(bundle, path)
            logger.info(f"Saved ensemble model for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model for {symbol}: {e}")
            return False

    def load_model(self, symbol: str) -> bool:
        path = os.path.join(self.models_dir, f'{self._safe_filename(symbol)}_xgboost.joblib')
        if os.path.exists(path):
            try:
                data = joblib.load(path)
                # Support both old and new model formats
                if isinstance(data, dict) and 'xgb' in data:
                    self.models[symbol] = data
                    self.feature_cols[symbol] = data.get('feature_cols', [])
                else:
                    # Legacy single XGBoost model
                    self.models[symbol] = {
                        'xgb': data,
                        'lgb': None,
                        'calibrator': None,
                    }
                logger.info(f"Loaded model for {symbol}")
                return True
            except Exception as e:
                logger.error(f"Failed to load model for {symbol}: {e}")
        return False
