from sklearn.linear_model import LinearRegression
from plantbrain_fastml.base.base_regressor import BaseRegressor
from optuna import Trial
import pandas as pd

class LinearRegressionRegressor(BaseRegressor):
    def __init__(self, **params):
        super().__init__(**params)
        self.model = LinearRegression(**params)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        preds = self.model.predict(X)
        return pd.Series(preds, index=X.index)

    def search_space(self, trial: Trial) -> dict:
        return {
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            # 'normalize' deprecated, no longer included
        }
