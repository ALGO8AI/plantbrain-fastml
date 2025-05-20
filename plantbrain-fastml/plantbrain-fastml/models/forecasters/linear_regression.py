import numpy as np
from sklearn.linear_model import LinearRegression
from plantbrain_fastml.base.base_forecaster import BaseForecaster
from optuna import Trial

class LinearRegressionForecaster(BaseForecaster):
    def __init__(self, **params):
        super().__init__(**params)
        self.model = LinearRegression(**params)
    
    def train(self, X, y):
        # X expected as time series features (e.g., lagged features)
        X = self.preprocessor.fit_transform(X)
        self.model.fit(X, y)
    
    def predict(self, X):
        X = self.preprocessor.transform(X)
        return self.model.predict(X)
    
    def search_space(self, trial: Trial):
        return {
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "normalize": trial.suggest_categorical("normalize", [True, False]),
        }
