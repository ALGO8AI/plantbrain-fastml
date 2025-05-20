from typing import List, Dict, Any

class ModelManagerMixin:
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def add_model(self, name: str, model):
        self.models[name] = model
    
    def train_all(self, X, y):
        for name, model in self.models.items():
            model.train(X, y)
    
    def evaluate_all(self, X, y, metrics=None) -> Dict[str, Dict[str, float]]:
        results = {}
        for name, model in self.models.items():
            results[name] = model.evaluate(X, y, metrics)
        self.results = results
        return results
    
    def hypertune_all(self, X, y, n_trials=20, timeout=None, metric=None, direction=None):
        for name, model in self.models.items():
            best_params = model.hypertune(X, y, n_trials=n_trials, timeout=timeout, metric=metric, direction=direction)
            print(f"[{name}] Best params: {best_params}")
    
    def get_best_model(self, metric, higher_is_better=True):
        best_name = None
        best_score = None
        for name, scores in self.results.items():
            score = scores.get(metric)
            if score is None:
                continue
            if best_score is None or (score > best_score if higher_is_better else score < best_score):
                best_score = score
                best_name = name
        return best_name, self.models[best_name]
