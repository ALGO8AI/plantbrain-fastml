# plantbrain_fastml/base/linear_regression_regressor.py

from sklearn.linear_model import LinearRegression
from plantbrain_fastml.base.base_regressor import BaseRegressor
from optuna import Trial
import pandas as pd

class LinearRegressionRegressor(BaseRegressor):
    """
    Linear Regression model with hyperparameter tuning and feature elimination / PCA support.
    """
    def __init__(self, **params):
        super().__init__(**params)
        self.model = LinearRegression(**params)

    def train(self, X, y,
              feature_elimination=False, fe_method=None, fe_n_features=None,
              pca=False, pca_n_components=None):
        X_processed = self._preprocess(X, y, feature_elimination, fe_method, fe_n_features, pca, pca_n_components, fit=True)
        self.model.fit(X_processed, y)

    def predict(self, X):
        X_processed = self.preprocessor.transform(X)

        if self.selected_features is not None:
            X_processed = pd.DataFrame(X_processed, index=X.index, columns=X.columns if hasattr(X, 'columns') else None)
            X_processed = X_processed[self.selected_features]

        if self.pca_model is not None:
            X_processed = self.pca_model.transform(X_processed)

        return self.model.predict(X_processed)

    def search_space(self, trial: Trial):
        # Simple linear regression params
        return {
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            # Note: 'normalize' param deprecated in latest sklearn versions
        }
