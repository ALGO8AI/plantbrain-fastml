# plantbrain-fastml/plantbrain_fastml/models/regressors/lasso.py

from sklearn.linear_model import Lasso as LassoModel
from plantbrain_fastml.base.base_regressor import BaseRegressor

class Lasso(BaseRegressor):
    def __init__(self, random_state=42, **kwargs):
        super().__init__() # <-- FIX: Removed **kwargs
        self.model = LassoModel(random_state=random_state, **kwargs)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def search_space(self, trial):
        return {
            "alpha": trial.suggest_float("alpha", 0.0001, 1.0, log=True),
        }