from sklearn.linear_model import LinearRegression
from plantbrain_fastml.base.base_regressor import BaseRegressor
from optuna import Trial

class LinearRegressionRegressor(BaseRegressor):
    def __init__(self, **params):
        super().__init__(**params)
        self.model = LinearRegression(**params)
    
    def train(self, X, y):
        X = self.preprocessor.fit_transform(X)
        self.model.fit(X, y)
    
    def predict(self, X):
        X = self.preprocessor.transform(X)
        return self.model.predict(X)
    
    def search_space(self, trial: Trial):
        # Linear regression has very few hyperparams, just fit_intercept as example
        return {
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "normalize": trial.suggest_categorical("normalize", [True, False]),
        }
