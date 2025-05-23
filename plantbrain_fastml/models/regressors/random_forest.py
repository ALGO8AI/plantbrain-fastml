from sklearn.ensemble import RandomForestRegressor
from plantbrain_fastml.base.base_regressor import BaseRegressor
from optuna import Trial
import pandas as pd

class RandomForestRegressorWrapper(BaseRegressor):
    def __init__(self, **params):
        super().__init__(**params)
        self.model = RandomForestRegressor(**params)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        preds = self.model.predict(X)
        return pd.Series(preds, index=X.index)

    def search_space(self, trial: Trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            # "max_features": trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "random_state": 42,
        }
