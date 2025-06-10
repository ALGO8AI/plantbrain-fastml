# plantbrain-fastml/plantbrain_fastml/models/regressors/bayesian_ridge.py

from sklearn.linear_model import BayesianRidge as BayesianRidgeModel
from plantbrain_fastml.base.base_regressor import BaseRegressor

class BayesianRidge(BaseRegressor):
    def __init__(self, **kwargs):
        super().__init__() # <-- FIX: Removed **kwargs
        self.model = BayesianRidgeModel(**kwargs)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def search_space(self, trial):
        return {
            "alpha_1": trial.suggest_float("alpha_1", 1e-7, 1e-5, log=True),
            "alpha_2": trial.suggest_float("alpha_2", 1e-7, 1e-5, log=True),
            "lambda_1": trial.suggest_float("lambda_1", 1e-7, 1e-5, log=True),
            "lambda_2": trial.suggest_float("lambda_2", 1e-7, 1e-5, log=True),
        }