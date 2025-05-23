from sklearn.base import BaseEstimator

class PreprocessWrapper(BaseEstimator):
    def __init__(self, base_model, outer_self, feature_elimination, fe_method, fe_n_features, pca, pca_n_components):
        self.base_model = base_model
        self.outer_self = outer_self  # reference to your regressor instance
        self.feature_elimination = feature_elimination
        self.fe_method = fe_method
        self.fe_n_features = fe_n_features
        self.pca = pca
        self.pca_n_components = pca_n_components

    def fit(self, X, y):
        X_processed = self.outer_self._preprocess(
            X, y,
            feature_elimination=self.feature_elimination,
            fe_method=self.fe_method,
            fe_n_features=self.fe_n_features,
            pca=self.pca,
            pca_n_components=self.pca_n_components,
            fit=True
        )
        self.base_model.fit(X_processed, y)
        return self

    def predict(self, X):
        X_processed = self.outer_self._preprocess(
            X, fit=False,
            feature_elimination=self.feature_elimination,
            fe_method=self.fe_method,
            fe_n_features=self.fe_n_features,
            pca=self.pca,
            pca_n_components=self.pca_n_components
        )
        return self.base_model.predict(X_processed)
