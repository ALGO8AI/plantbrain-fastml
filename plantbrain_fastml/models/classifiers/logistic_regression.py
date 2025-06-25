# plantbrain-fastml/plantbrain_fastml/models/classifiers/logistic_regression.py

from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from ...base.base_classifier import BaseClassifier
from optuna import Trial

class LogisticRegression(BaseClassifier):
    def __init__(self, **params):
        super().__init__(**params)
        if 'random_state' not in self.params:
            self.params['random_state'] = 42
        if 'max_iter' not in self.params:
            self.params['max_iter'] = 1000
        self.model = SklearnLogisticRegression(**self.params)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def search_space(self, trial: Trial) -> dict:
        """
        Defines the conditional hyperparameter space for Optuna.
        """
        params = {}
        params['solver'] = trial.suggest_categorical("solver", ["liblinear", "saga"])

        if params['solver'] == 'liblinear':
            params['penalty'] = trial.suggest_categorical("penalty_for_liblinear", ["l1", "l2"])
        elif params['solver'] == 'saga':
            penalty_choice = trial.suggest_categorical("penalty_for_saga", ["l1", "l2", "elasticnet"])
            params['penalty'] = penalty_choice
            if penalty_choice == 'elasticnet':
                params['l1_ratio'] = trial.suggest_float("l1_ratio_for_saga", 0.0, 1.0)

        params['C'] = trial.suggest_float('C', 1e-4, 1e2, log=True)
        params['max_iter'] = trial.suggest_int("max_iter", 1000, 4000)
        params['random_state'] = 42

        return params