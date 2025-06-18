# plantbrain-fastml/plantbrain_fastml/models/classifiers/svc.py
from sklearn.svm import SVC
from ...base.base_classifier import BaseClassifier
from optuna import Trial

class SVCWrapper(BaseClassifier):
    def __init__(self, **params):
        # probability=True is required for predict_proba
        if 'probability' not in params:
            params['probability'] = True
        super().__init__(**params)
        self.model = SVC(**params, random_state=42)

    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def search_space(self, trial: Trial):
        return {
            'C': trial.suggest_float('C', 1e-2, 1e2, log=True),
            'gamma': trial.suggest_float('gamma', 1e-4, 1e-1, log=True),
            'kernel': trial.suggest_categorical('kernel', ['rbf'])
        }