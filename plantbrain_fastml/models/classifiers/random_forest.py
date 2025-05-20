from sklearn.ensemble import RandomForestClassifier
from plantbrain_fastml.base.base_classifier import BaseClassifier
from optuna import Trial

class RandomForestClassifierModel(BaseClassifier):
    def __init__(self, **params):
        super().__init__(**params)
        self.model = RandomForestClassifier(**params)
    
    def train(self, X, y):
        X = self.preprocessor.fit_transform(X)
        self.model.fit(X, y)
    
    def predict(self, X):
        X = self.preprocessor.transform(X)
        return self.model.predict(X)
    
    def search_space(self, trial: Trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
        }
