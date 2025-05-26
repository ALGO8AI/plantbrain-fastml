from sklearn.svm import SVR
from plantbrain_fastml.base.base_regressor import BaseRegressor
from optuna import Trial
import pandas as pd

class SVRWrapper(BaseRegressor):
    def __init__(self, **params):
        super().__init__(**params)
        self.model = SVR(**params)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        preds = self.model.predict(X)
        return pd.Series(preds, index=X.index)

    def search_space(self, trial: Trial) -> dict:
        return {
            "kernel": trial.suggest_categorical("kernel", ["rbf", "linear", "poly", "sigmoid"]),
            "C": trial.suggest_float("C", 0.1, 100.0, log=True),
            "epsilon": trial.suggest_float("epsilon", 0.01, 1.0),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }
