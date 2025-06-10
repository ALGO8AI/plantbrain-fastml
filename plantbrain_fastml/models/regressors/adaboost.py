# plantbrain-fastml/plantbrain_fastml/models/regressors/adaboost.py

from sklearn.ensemble import AdaBoostRegressor
from plantbrain_fastml.base.base_regressor import BaseRegressor

class AdaBoost(BaseRegressor):
    def __init__(self, random_state=42, **kwargs):
        super().__init__() # <-- FIX: Removed **kwargs
        self.model = AdaBoostRegressor(random_state=random_state, **kwargs)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def search_space(self, trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 1.0, log=True),
            "loss": trial.suggest_categorical("loss", ["linear", "square", "exponential"]),
        }