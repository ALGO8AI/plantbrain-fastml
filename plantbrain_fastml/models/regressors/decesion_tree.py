from sklearn.tree import DecisionTreeRegressor
from plantbrain_fastml.base.base_regressor import BaseRegressor
from optuna import Trial
import pandas as pd
class DecisionTreeRegressorModel(BaseRegressor):
    def __init__(self, **params):
        super().__init__(**params)
        self.model = DecisionTreeRegressor(**params)

    def train(self, X, y,
              feature_elimination=False, fe_method=None, fe_n_features=None,
              pca=False, pca_n_components=None):
        X_processed = self._preprocess(X, y, feature_elimination, fe_method, fe_n_features, pca, pca_n_components, fit=True)
        self.model.fit(X_processed, y)

    def predict(self, X):
        X_processed = self.preprocessor.transform(X)
        if self.selected_features is not None:
            X_processed = pd.DataFrame(X_processed, index=getattr(X, 'index', None), columns=getattr(X, 'columns', None))
            X_processed = X_processed[self.selected_features]
        if self.pca_model is not None:
            X_processed = self.pca_model.transform(X_processed)
        return self.model.predict(X_processed)

    def search_space(self, trial: Trial):
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
        }
