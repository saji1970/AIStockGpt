"""
Optuna-based hyperparameter optimization for XGBoost + LightGBM ensemble.
Uses walk-forward cross-validation to avoid look-ahead bias.
Saves best params per symbol for reuse.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, f1_score

logger = logging.getLogger(__name__)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.info("Optuna not available, using default hyperparameters")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class HyperparamTuner:
    """Optuna hyperparameter tuner with walk-forward CV objective."""

    EXCLUDED_COLS = frozenset([
        'open', 'high', 'low', 'close', 'volume',
        'dividends', 'stock_splits', 'capital_gains',
    ])

    def __init__(self, params_dir: str = 'models'):
        self.params_dir = params_dir
        os.makedirs(params_dir, exist_ok=True)

    def tune(
        self,
        symbol: str,
        features_df: pd.DataFrame,
        target: np.ndarray,
        n_trials: int = 30,
        timeout: int = 300,
    ) -> Dict[str, Dict]:
        """
        Run Optuna optimization for both XGBoost and LightGBM.
        Returns dict with 'xgb_params' and 'lgb_params'.
        """
        if not OPTUNA_AVAILABLE:
            logger.info(f"Optuna not available, returning defaults for {symbol}")
            return self._default_params()

        feature_cols = [c for c in features_df.columns if c not in self.EXCLUDED_COLS]
        X = features_df[feature_cols].values.astype(np.float32)
        y = target

        # Remove NaN targets
        valid = ~np.isnan(y)
        X, y = X[valid], y[valid]

        if len(X) < 200:
            logger.info(f"Insufficient data for tuning {symbol} ({len(X)} rows), using defaults")
            return self._default_params()

        # XGBoost tuning
        logger.info(f"Tuning XGBoost for {symbol} ({n_trials} trials)...")
        xgb_study = optuna.create_study(direction='minimize', study_name=f'{symbol}_xgb')
        xgb_study.optimize(
            lambda trial: self._xgb_objective(trial, X, y),
            n_trials=n_trials,
            timeout=timeout // 2,
            show_progress_bar=False,
        )
        xgb_best = self._xgb_trial_to_params(xgb_study.best_trial)
        logger.info(f"  XGBoost best loss: {xgb_study.best_value:.4f}")

        # LightGBM tuning
        lgb_best = {}
        if LIGHTGBM_AVAILABLE:
            logger.info(f"Tuning LightGBM for {symbol} ({n_trials} trials)...")
            lgb_study = optuna.create_study(direction='minimize', study_name=f'{symbol}_lgb')
            lgb_study.optimize(
                lambda trial: self._lgb_objective(trial, X, y),
                n_trials=n_trials,
                timeout=timeout // 2,
                show_progress_bar=False,
            )
            lgb_best = self._lgb_trial_to_params(lgb_study.best_trial)
            logger.info(f"  LightGBM best loss: {lgb_study.best_value:.4f}")

        result = {'xgb_params': xgb_best, 'lgb_params': lgb_best}

        # Save best params
        self._save_params(symbol, result)

        return result

    def load_params(self, symbol: str) -> Optional[Dict]:
        """Load previously tuned params for a symbol."""
        path = os.path.join(self.params_dir, f'{symbol}_hparams.json')
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load params for {symbol}: {e}")
        return None

    # ── Objectives ────────────────────────────────────────────

    def _xgb_objective(self, trial: 'optuna.Trial', X: np.ndarray, y: np.ndarray) -> float:
        """Walk-forward CV objective for XGBoost."""
        params = {
            'n_estimators': 500,
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 30),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),

            'eval_metric': 'logloss',
            'early_stopping_rounds': 30,
            'random_state': 42,
            'n_jobs': -1,
        }

        return self._walk_forward_cv(X, y, 'xgb', params)

    def _lgb_objective(self, trial: 'optuna.Trial', X: np.ndarray, y: np.ndarray) -> float:
        """Walk-forward CV objective for LightGBM."""
        params = {
            'n_estimators': 500,
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
        }

        return self._walk_forward_cv(X, y, 'lgb', params)

    def _walk_forward_cv(
        self, X: np.ndarray, y: np.ndarray, model_type: str, params: Dict
    ) -> float:
        """3-fold walk-forward validation returning average log loss."""
        n = len(X)
        train_size = int(n * 0.5)
        fold_size = (n - train_size) // 3

        losses = []
        for fold in range(3):
            train_end = train_size + fold * fold_size
            test_end = min(train_end + fold_size, n)

            if test_end <= train_end:
                continue

            X_train, y_train = X[:train_end], y[:train_end]
            X_test, y_test = X[train_end:test_end], y[train_end:test_end]

            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            try:
                if model_type == 'xgb':
                    model = XGBClassifier(**params)
                    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
                else:
                    model = lgb.LGBMClassifier(**params)
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_test, y_test)],
                        callbacks=[lgb.log_evaluation(0), lgb.early_stopping(20, verbose=False)],
                    )

                proba = model.predict_proba(X_test)
                loss = log_loss(y_test, proba, labels=[0, 1])
                losses.append(loss)
            except Exception:
                losses.append(1.0)  # penalty for failed fold

        return float(np.mean(losses)) if losses else 1.0

    # ── Param conversion ──────────────────────────────────────

    def _xgb_trial_to_params(self, trial) -> Dict:
        """Convert Optuna trial to XGBoost param dict."""
        return {
            'n_estimators': 500,
            'max_depth': trial.params['max_depth'],
            'learning_rate': trial.params['learning_rate'],
            'subsample': trial.params['subsample'],
            'colsample_bytree': trial.params['colsample_bytree'],
            'min_child_weight': trial.params['min_child_weight'],
            'reg_alpha': trial.params['reg_alpha'],
            'reg_lambda': trial.params['reg_lambda'],
            'gamma': trial.params['gamma'],

            'eval_metric': 'logloss',
            'early_stopping_rounds': 30,
            'random_state': 42,
            'n_jobs': -1,
        }

    def _lgb_trial_to_params(self, trial) -> Dict:
        """Convert Optuna trial to LightGBM param dict."""
        return {
            'n_estimators': 500,
            'max_depth': trial.params['max_depth'],
            'learning_rate': trial.params['learning_rate'],
            'subsample': trial.params['subsample'],
            'colsample_bytree': trial.params['colsample_bytree'],
            'min_child_samples': trial.params['min_child_samples'],
            'reg_alpha': trial.params['reg_alpha'],
            'reg_lambda': trial.params['reg_lambda'],
            'num_leaves': trial.params['num_leaves'],
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
        }

    def _default_params(self) -> Dict:
        """Return default params when Optuna is not available."""
        return {
            'xgb_params': {
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
            },
            'lgb_params': {
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
            },
        }

    def _save_params(self, symbol: str, params: Dict) -> None:
        """Save tuned params to JSON."""
        path = os.path.join(self.params_dir, f'{symbol}_hparams.json')
        try:
            with open(path, 'w') as f:
                json.dump(params, f, indent=2)
            logger.info(f"Saved tuned hyperparams for {symbol}")
        except Exception as e:
            logger.warning(f"Failed to save params for {symbol}: {e}")
