from sklearn.neighbors import KNeighborsRegressor
from plantbrain_fastml.base.base_regressor import BaseRegressor
from optuna import Trial
import pandas as pd

class KNeighborsRegressorWrapper(BaseRegressor):
    def __init__(self, **params):
        super().__init__(**params)
        self.model = KNeighborsRegressor(**params)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        preds = self.model.predict(X)
        return pd.Series(preds, index=X.index)

    def search_space(self, trial: Trial) -> dict:
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 20),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "algorithm": trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"]),
            "leaf_size": trial.suggest_int("leaf_size", 20, 100),
            "p": trial.suggest_int("p", 1, 2),  # 1=Manhattan, 2=Euclidean
        }
