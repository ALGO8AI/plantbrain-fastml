from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import optuna
from plantbrain_fastml.utils.preprocessing import default_preprocessor
from plantbrain_fastml.utils.metrics import classification_metrics

class BaseClassifier(ABC):
    def __init__(self, **params):
        self.params = params
        self.model = None
        self.preprocessor = default_preprocessor()
    
    @abstractmethod
    def train(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    def evaluate(self, X, y, metrics=None) -> Dict[str, float]:
        if metrics is None:
            metrics = classification_metrics
        y_pred = self.predict(X)
        results = {name: fn(y, y_pred) for name, fn in metrics.items()}
        return results
    
    def set_params(self, **params):
        self.params.update(params)
    
    def hypertune(self, X, y, n_trials=20, timeout=None, metric='accuracy', direction='maximize'):
        def objective(trial):
            search_space = self.search_space(trial)
            self.set_params(**search_space)
            self.train(X, y)
            evals = self.evaluate(X, y)
            return evals[metric]
        
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        self.set_params(**study.best_params)
        self.train(X, y)
        return study.best_params
    
    @abstractmethod
    def search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        pass
