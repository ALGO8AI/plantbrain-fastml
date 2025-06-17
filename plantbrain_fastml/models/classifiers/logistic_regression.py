# plantbrain-fastml/plantbrain_fastml/models/classifiers/logistic_regression.py
from sklearn.linear_model import LogisticRegression
from ...base.base_classifier import BaseClassifier
from optuna import Trial

class LogisticRegressionWrapper(BaseClassifier):
    def __init__(self, **params):
        super().__init__(**params)
        self.model = LogisticRegression(**params, random_state=42, max_iter=1000)

    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def search_space(self, trial: Trial):
        return {
            'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'saga'])
        }