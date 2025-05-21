# plantbrain_fastml/base/base_regressor.py

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate, train_test_split
from plantbrain_fastml.utils.preprocessing import default_preprocessor
from plantbrain_fastml.utils.metrics import regression_metrics
import optuna


class BaseRegressor(ABC):
    """
    Abstract base class for regressors with built-in feature elimination,
    dimensionality reduction (PCA), CV or train/test evaluation, and hypertuning support.

    Feature elimination methods supported:
      - 'lasso': Lasso based selection
      - 'tree': Tree-based feature importance (Decision Tree or Random Forest)
      - 'correlation': Correlation based selection of top N features
      - None: no feature elimination

    Dimensionality reduction:
      - PCA only (optional)

    Evaluation supports:
      - cross-validation (cv_folds > 1)
      - train-test split (cv_folds = 1)
    """

    def __init__(self, **params):
        """
        Initialize the base regressor.

        Parameters:
        -----------
        params : dict
            Model-specific parameters.
        """
        self.params = params
        self.model = None
        self.preprocessor = default_preprocessor()
        self.selected_features = None  # Stores columns selected after feature elimination
        self.pca_model = None          # PCA model if applied

    @abstractmethod
    def train(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> None:
        """
        Train the model on training data X, y.
        Must be implemented by subclasses.

        Parameters:
        -----------
        X : pd.DataFrame
            Training features.
        y : array-like
            Training target.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict target values for input features X.
        Must be implemented by subclasses.

        Parameters:
        -----------
        X : pd.DataFrame
            Input features.

        Returns:
        --------
        np.ndarray
            Predicted values.
        """
        pass

    def _feature_elimination(self,
                             X: pd.DataFrame,
                             y: Union[pd.Series, np.ndarray],
                             method: Optional[str],
                             n_features: Optional[int]) -> pd.DataFrame:
        """
        Perform feature elimination on X.

        Parameters:
        -----------
        X : pd.DataFrame
            Input features.
        y : array-like
            Target values.
        method : str or None
            Feature elimination method: 'lasso', 'tree', 'correlation', or None.
        n_features : int or None
            Number of features to select. If None, select all features.

        Returns:
        --------
        pd.DataFrame
            DataFrame with selected features only.
        """
        if method is None:
            self.selected_features = X.columns.tolist()
            return X

        if method == 'lasso':
            # Use LassoCV for feature selection
            lasso = LassoCV(cv=5, random_state=42, n_jobs=-1).fit(X, y)
            selector = SelectFromModel(lasso, prefit=True, max_features=n_features)
            mask = selector.get_support()
            features_selected = X.columns[mask]
            if n_features is not None and len(features_selected) > n_features:
                # Select top n_features by absolute coefficient magnitude
                coef_abs = np.abs(lasso.coef_[mask])
                top_idx = np.argsort(coef_abs)[-n_features:]
                features_selected = features_selected[top_idx]
            self.selected_features = list(features_selected)
            return X[self.selected_features]

        elif method == 'tree':
            # Use RandomForestRegressor for feature importance
            tree_model = RandomForestRegressor(random_state=42, n_jobs=-1)
            tree_model.fit(X, y)
            importances = tree_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            if n_features is not None:
                indices = indices[:n_features]
            features_selected = X.columns[indices]
            self.selected_features = list(features_selected)
            return X[self.selected_features]

        elif method == 'correlation':
            # Select top n_features by absolute correlation with target
            corrs = X.apply(lambda col: np.abs(np.corrcoef(col, y)[0, 1]))
            corrs = corrs.fillna(0)
            selected = corrs.sort_values(ascending=False)
            if n_features is not None:
                selected = selected.iloc[:n_features]
            self.selected_features = list(selected.index)
            return X[self.selected_features]

        else:
            raise ValueError(f"Unknown feature elimination method: {method}")

    def _dimensionality_reduction(self,
                                  X: pd.DataFrame,
                                  n_components: Optional[int]) -> pd.DataFrame:
        """
        Perform PCA on features X.

        Parameters:
        -----------
        X : pd.DataFrame
            Input features (after feature elimination).
        n_components : int or None
            Number of PCA components to keep. If None or >= n_features, no PCA is applied.

        Returns:
        --------
        pd.DataFrame
            Transformed features after PCA or original if not applied.
        """
        if n_components is None or n_components >= X.shape[1]:
            self.pca_model = None
            return X

        self.pca_model = PCA(n_components=n_components, random_state=42)
        X_reduced = self.pca_model.fit_transform(X)
        columns = [f'pca_{i + 1}' for i in range(X_reduced.shape[1])]
        return pd.DataFrame(X_reduced, columns=columns, index=X.index)

    def _preprocess(self,
                    X: pd.DataFrame,
                    y: Optional[Union[pd.Series, np.ndarray]] = None,
                    feature_elimination: bool = False,
                    fe_method: Optional[str] = None,
                    fe_n_features: Optional[int] = None,
                    pca: bool = False,
                    pca_n_components: Optional[int] = None,
                    fit: bool = True) -> pd.DataFrame:
        """
        Full preprocessing pipeline: preprocessing, feature elimination, PCA.

        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Input features.
        y : array-like or None
            Target variable (required if feature elimination is True).
        feature_elimination : bool
            Whether to apply feature elimination.
        fe_method : str or None
            Feature elimination method ('lasso', 'tree', 'correlation').
        fe_n_features : int or None
            Number of features to select in feature elimination.
        pca : bool
            Whether to apply PCA.
        pca_n_components : int or None
            Number of PCA components.
        fit : bool
            If True, fit preprocessors; else transform only.

        Returns:
        --------
        pd.DataFrame
            Preprocessed feature dataframe.
        """
        if fit:
            X_processed = self.preprocessor.fit_transform(X)
        else:
            X_processed = self.preprocessor.transform(X)

        # Convert to DataFrame for feature selection
        if not isinstance(X_processed, pd.DataFrame):
            idx = getattr(X, 'index', None)
            cols = getattr(X, 'columns', None)
            X_processed = pd.DataFrame(X_processed, index=idx, columns=cols)

        if feature_elimination and y is not None:
            X_processed = self._feature_elimination(X_processed, y, fe_method, fe_n_features)
        elif not feature_elimination:
            # Reset selected features if no FE
            self.selected_features = X_processed.columns.tolist()

        if pca:
            X_processed = self._dimensionality_reduction(X_processed, pca_n_components)
        else:
            self.pca_model = None

        return X_processed

    def train(self,
              X: pd.DataFrame,
              y: Union[pd.Series, np.ndarray],
              feature_elimination: bool = False,
              fe_method: Optional[str] = None,
              fe_n_features: Optional[int] = None,
              pca: bool = False,
              pca_n_components: Optional[int] = None) -> None:
        """
        Train the model with optional feature elimination and PCA.

        Parameters:
        -----------
        X : pd.DataFrame
            Training features.
        y : array-like
            Training targets.
        feature_elimination : bool
            Whether to perform feature elimination.
        fe_method : str or None
            Feature elimination method ('lasso', 'tree', 'correlation').
        fe_n_features : int or None
            Number of features to select.
        pca : bool
            Whether to apply PCA.
        pca_n_components : int or None
            Number of PCA components.
        """
        X_processed = self._preprocess(X, y, feature_elimination, fe_method, fe_n_features, pca, pca_n_components, fit=True)
        self.model.fit(X_processed, y)

    def predict(self,
                X: pd.DataFrame) -> np.ndarray:
        """
        Predict using the trained model.

        Parameters:
        -----------
        X : pd.DataFrame
            Input features.

        Returns:
        --------
        np.ndarray
            Predicted values.
        """
        X_processed = self.preprocessor.transform(X)

        # Apply feature selection if done
        if self.selected_features is not None:
            X_processed = pd.DataFrame(X_processed, index=X.index, columns=X.columns if hasattr(X, 'columns') else None)
            # Select the features used for training
            X_processed = X_processed[self.selected_features]

        # Apply PCA if done
        if self.pca_model is not None:
            X_processed = self.pca_model.transform(X_processed)

        return self.model.predict(X_processed)

    def evaluate(self,
                 X: pd.DataFrame,
                 y: Union[pd.Series, np.ndarray],
                 metrics: Optional[Dict[str, Any]] = None,
                 cv_folds: int = 5,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 feature_elimination: bool = False,
                 fe_method: Optional[str] = None,
                 fe_n_features: Optional[int] = None,
                 pca: bool = False,
                 pca_n_components: Optional[int] = None,
                 return_cv_scores: bool = False) -> Union[Dict[str, float], pd.DataFrame]:
        """
        Evaluate model performance via cross-validation or train/test split.

        Parameters:
        -----------
        X : pd.DataFrame
            Input features.
        y : array-like
            Target values.
        metrics : dict or None
            Metrics dictionary, keys are metric names, values are functions(y_true, y_pred).
            Defaults to regression_metrics.
        cv_folds : int
            Number of CV folds. If 1, use train-test split.
        test_size : float
            Proportion of test data if train-test split is used.
        random_state : int
            Random seed for reproducibility.
        feature_elimination : bool
            Whether to apply feature elimination during training.
        fe_method : str or None
            Feature elimination method.
        fe_n_features : int or None
            Number of features for feature elimination.
        pca : bool
            Whether to apply PCA.
        pca_n_components : int or None
            Number of PCA components.
        return_cv_scores : bool
            If True and cv_folds>1, return detailed CV scores dataframe.

        Returns:
        --------
        Dict[str, float] or pd.DataFrame
            Dictionary of averaged metric scores or detailed CV scores DataFrame if return_cv_scores=True.
        """
        if metrics is None:
            metrics = regression_metrics

        if cv_folds > 1:
            # Cross-validation evaluation
            scoring = {name: 'neg_root_mean_squared_error' if name == 'rmse' else
                             'r2' if name == 'r2' else
                             'neg_mean_absolute_error' if name == 'mae' else None
                       for name in metrics.keys()}
            # Adjust for sklearn naming and negative metrics
            # cross_validate expects scoring keys from sklearn or callables - so we define mapping here:
            scoring_map = {}
            for name in metrics.keys():
                if name == 'rmse':
                    # sklearn has neg_root_mean_squared_error only since v0.22; if not available fallback to neg_mean_squared_error
                    scoring_map[name] = 'neg_root_mean_squared_error'
                elif name == 'r2':
                    scoring_map[name] = 'r2'
                elif name == 'mae':
                    scoring_map[name] = 'neg_mean_absolute_error'
                else:
                    scoring_map[name] = None  # unsupported

            # Prepare a wrapper estimator with preprocessing integrated
            def pipeline_fit(X_cv, y_cv):
                self.train(X_cv, y_cv, feature_elimination, fe_method, fe_n_features, pca, pca_n_components)

            def pipeline_predict(X_cv):
                return self.predict(X_cv)

            # Use cross_validate with custom scoring functions
            cv_results = cross_validate(self.model, X, y, cv=cv_folds, scoring=scoring_map, n_jobs=-1,
                                         return_train_score=False)

            # Calculate metrics manually since cross_validate returns negative metrics for some scores
            scores = {}
            for metric_name in metrics.keys():
                key = f'test_{metric_name}'
                if metric_name == 'rmse':
                    # sklearn neg_root_mean_squared_error is negative, so convert back
                    scores[metric_name] = -np.mean(cv_results[f'test_rmse'])
                elif metric_name == 'mae':
                    scores[metric_name] = -np.mean(cv_results[f'test_mae'])
                elif metric_name == 'r2':
                    scores[metric_name] = np.mean(cv_results[f'test_r2'])
                else:
                    scores[metric_name] = np.nan

            if return_cv_scores:
                # Provide full cv results as DataFrame
                df = pd.DataFrame({k: -v if k.startswith('test_rmse') or k.startswith('test_mae') else v for k, v in cv_results.items()})
                return df

            return scores

        else:
            # Train-test split evaluation
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            self.train(X_train, y_train, feature_elimination, fe_method, fe_n_features, pca, pca_n_components)
            y_pred = self.predict(X_test)
            scores = {name: fn(y_test, y_pred) for name, fn in metrics.items()}
            return scores

    def hypertune(self,
                  X: pd.DataFrame,
                  y: Union[pd.Series, np.ndarray],
                  n_trials: int = 20,
                  timeout: Optional[int] = None,
                  metric: str = 'rmse',
                  direction: str = 'minimize',
                  feature_elimination: bool = False,
                  fe_method: Optional[str] = None,
                  fe_n_features: Optional[int] = None,
                  pca: bool = False,
                  pca_n_components: Optional[int] = None,
                  cv_folds: int = 5,
                  test_size: float = 0.2,
                  random_state: int = 42) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using Optuna.

        Parameters:
        -----------
        X : pd.DataFrame
            Input features.
        y : array-like
            Target variable.
        n_trials : int
            Number of trials for Optuna.
        timeout : int or None
            Timeout in seconds for Optuna study.
        metric : str
            Metric to optimize, key from regression_metrics.
        direction : str
            'minimize' or 'maximize'.
        feature_elimination : bool
            Whether to apply feature elimination during tuning.
        fe_method : str or None
            Feature elimination method.
        fe_n_features : int or None
            Number of features for feature elimination.
        pca : bool
            Whether to apply PCA during tuning.
        pca_n_components : int or None
            Number of PCA components.
        cv_folds : int
            Number of CV folds for evaluation during tuning.
        test_size : float
            Test size if using train-test split.
        random_state : int
            Random seed.

        Returns:
        --------
        Dict[str, Any]
            Best parameters found by Optuna.
        """
        def objective(trial: optuna.Trial) -> float:
            # Get suggested params from child class
            trial_params = self.search_space(trial)
            self.set_params(**trial_params)

            # Train and evaluate using CV
            evals = self.evaluate(X, y, metrics=regression_metrics,
                                  cv_folds=cv_folds,
                                  test_size=test_size,
                                  feature_elimination=feature_elimination,
                                  fe_method=fe_method,
                                  fe_n_features=fe_n_features,
                                  pca=pca,
                                  pca_n_components=pca_n_components,
                                  return_cv_scores=False)

            return evals[metric]

        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        self.set_params(**study.best_params)
        self.train(X, y, feature_elimination, fe_method, fe_n_features, pca, pca_n_components)

        return study.best_params

    def set_params(self, **params):
        """
        Update model parameters.

        Parameters:
        -----------
        params : dict
            Parameters to update.
        """
        self.params.update(params)
        if self.model is not None:
            self.model.set_params(**params)

    @abstractmethod
    def search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define hyperparameter search space for Optuna.

        Parameters:
        -----------
        trial : optuna.Trial
            Trial object to suggest parameters.

        Returns:
        --------
        dict
            Dictionary of parameters for the model.
        """
        pass
