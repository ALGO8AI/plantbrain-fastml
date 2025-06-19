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
        params = {}
        
        # 1. Suggest a kernel
        kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
        params["kernel"] = kernel

        # 2. Add general parameters
        params["C"] = trial.suggest_float("C", 1e-2, 1e3, log=True)

        # 3. Add parameters conditional on the kernel choice
        if kernel == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 5)
            params["coef0"] = trial.suggest_float("coef0", 0.0, 10.0)
        
        if kernel in ["rbf", "poly", "sigmoid"]:
            params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
            
        params["probability"] = True # Keep this fixed
        params["random_state"] = 42
        return params