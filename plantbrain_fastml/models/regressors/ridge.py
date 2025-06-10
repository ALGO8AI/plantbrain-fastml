# plantbrain-fastml/plantbrain_fastml/models/regressors/ridge.py

from sklearn.linear_model import Ridge as RidgeModel
from plantbrain_fastml.base.base_regressor import BaseRegressor

class Ridge(BaseRegressor):
    def __init__(self, random_state=42, **kwargs):
        super().__init__() # <-- FIX: Removed **kwargs
        self.model = RidgeModel(random_state=random_state, **kwargs)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def search_space(self, trial):
        return {
            "alpha": trial.suggest_float("alpha", 0.0001, 1.0, log=True),
        }