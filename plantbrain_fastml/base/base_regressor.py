# plantbrain_fastml/base/base_regressor.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
import optuna
from plantbrain_fastml.base.base_preprocessor import BasePreprocessor
from plantbrain_fastml.utils.metrics import regression_metrics

class BaseRegressor(ABC):
    """
    Abstract base class for regression models with:
    - Preprocessing: cleaning, feature elimination, PCA (via BasePreprocessor)
    - Cross-validation training on train split + test set evaluation
    - Internal hyperparameter tuning support with Optuna in evaluate()
    - Test set plots (line and scatter)
    """

    def __init__(self, **params):
        self.params = params
        self.model = None
        self.preprocessor = BasePreprocessor()
        self.cv_results = None
        self.test_results = None

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        pass

    def set_params(self, **params):
        self.params.update(params)
        if self.model is not None:
            self.model.set_params(**params)

    def evaluate(self,
                 X: pd.DataFrame,
                 y: Union[pd.Series, np.ndarray],
                 metrics: Optional[Dict[str, Any]] = None,
                 cv_folds: int = 5,
                 test_size: float = 0.2,
                 test_split_type: str = 'random',
                 feature_elimination: bool = False,
                 fe_method: Optional[str] = None,
                 fe_n_features: Optional[int] = None,
                 pca: bool = False,
                 pca_n_components: Optional[int] = None,
                 hypertune: bool = False,
                 hypertune_params: Optional[Dict[str, Any]] = None,
                 hypertune_metrics: str = 'rmse',
                 return_plots: bool = True,
                 random_state: int = 42
                 ) -> Dict[str, Any]:
        """
        Evaluate the model with optional hypertuning inside.
        Performs train-test split, CV on train split, and test evaluation.

        Parameters:
        - hypertune (bool): If True, perform hyperparameter tuning before training.
        - hypertune_params (dict): Passed to hypertune method.
        """

        if metrics is None:
            metrics = regression_metrics

        hypertune_params = hypertune_params or {}

        # Step 1: Split data
        if test_split_type == 'random':
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=True, random_state=random_state)
        elif test_split_type == 'ordered':
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        else:
            raise ValueError("test_split_type must be 'random' or 'ordered'")

        # Step 2: Initialize preprocessor and fit on training data
        self.preprocessor = BasePreprocessor(
            feature_elimination=feature_elimination,
            fe_method=fe_method,
            fe_n_features=fe_n_features,
            pca=pca,
            pca_n_components=pca_n_components)

        X_train_proc, y_train_proc = self.preprocessor.fit_transform(X_train, y_train)

        # Step 3: Optional hypertuning on training split
        if hypertune:

            if hypertune_metrics not in metrics:
                raise ValueError(f"Not a valid metric: {hypertune_metrics}")
            if 'r2' in hypertune_metrics or 'neg' in hypertune_metrics:
                direction= "maximize"
            else:
                direction= "minimize"
            def objective(trial: optuna.Trial) -> float:
                trial_params = self.search_space(trial)
                self.set_params(**trial_params)
                # Evaluate CV score on training split only (no test)
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                cv_scores = []
                for train_idx, val_idx in cv.split(X_train_proc):
                    X_cv_train, X_cv_val = X_train_proc.iloc[train_idx], X_train_proc.iloc[val_idx]
                    y_cv_train, y_cv_val = y_train_proc.iloc[train_idx], y_train_proc.iloc[val_idx]
                    self.train(X_cv_train, y_cv_train)
                    y_pred_val = self.predict(X_cv_val)
                    score = metrics[hypertune_metrics](y_cv_val, y_pred_val)  # default RMSE if available
                    cv_scores.append(score)
                return np.mean(cv_scores)

            # Suppress Optuna logs for clean output
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            study = optuna.create_study(direction=direction)
            study.optimize(objective, n_trials=hypertune_params.get('n_trials', 20),
                           timeout=hypertune_params.get('timeout', None))
            self.set_params(**study.best_params)

        # Step 4: CV on training split with chosen params
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_scores = {name: [] for name in metrics.keys()}
        for train_idx, val_idx in cv.split(X_train_proc):
            X_cv_train, X_cv_val = X_train_proc.iloc[train_idx], X_train_proc.iloc[val_idx]
            y_cv_train, y_cv_val = y_train_proc.iloc[train_idx], y_train_proc.iloc[val_idx]
            self.train(X_cv_train, y_cv_train)
            y_pred_val = self.predict(X_cv_val)
            for name, metric_fn in metrics.items():
                score = metric_fn(y_cv_val, y_pred_val)
                cv_scores[name].append(score)
        cv_scores_summary = {name: (np.mean(scores), np.std(scores)) for name, scores in cv_scores.items()}

        # Step 5: Preprocess and evaluate on test split
        X_test_proc = self.preprocessor.transform(X_test)
        y_test_clean = y_test.dropna()
        common_idx = X_test_proc.index.intersection(y_test_clean.index)
        X_test_proc = X_test_proc.loc[common_idx]
        y_test_clean = y_test_clean.loc[common_idx]

        self.train(X_train_proc, y_train_proc)
        y_pred_test = self.predict(X_test_proc)

        test_scores = {name: fn(y_test_clean, y_pred_test) for name, fn in metrics.items()}

        # Step 6: Generate plots
        plots = {}
        if return_plots:
            plots['line'] = self._plot_line(y_test_clean, y_pred_test)
            plots['scatter'] = self._plot_scatter(y_test_clean, y_pred_test)
            self.latest_plots = plots

        self.cv_results = cv_scores_summary
        self.test_results = test_scores
        

        return {
            'cv_scores': cv_scores_summary,
            'test_scores': test_scores,
            'plots': plots
        }

    def _plot_line(self, y_true: pd.Series, y_pred: np.ndarray):
        fig, ax = plt.subplots()
        ax.plot(y_true.index, y_true.values, label='True')
        ax.plot(y_true.index, y_pred, label='Predicted')
        ax.set_title("Line Plot: True vs Predicted")
        ax.legend()
        plt.close(fig)
        return fig

    def _plot_scatter(self, y_true: pd.Series, y_pred: np.ndarray):
        fig, ax = plt.subplots()
        ax.scatter(y_true, y_pred, alpha=0.5)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Scatter Plot: True vs Predicted")
        plt.close(fig)
        return fig
