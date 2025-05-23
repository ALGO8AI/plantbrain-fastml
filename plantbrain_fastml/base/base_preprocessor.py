# plantbrain_fastml/base/preprocessor.py
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin

class BasePreprocessor(BaseEstimator, TransformerMixin):
    """
    Base preprocessor with cleaning, feature elimination, and PCA.
    Designed to be subclassed for different model types.
    """
    def __init__(self,
                 feature_elimination: bool = False,
                 fe_method: Optional[str] = None,
                 fe_n_features: Optional[int] = None,
                 pca: bool = False,
                 pca_n_components: Optional[int] = None):
        self.feature_elimination = feature_elimination
        self.fe_method = fe_method
        self.fe_n_features = fe_n_features
        self.pca = pca
        self.pca_n_components = pca_n_components
        
        self.selected_features = None
        self.pca_model = None

    def fit(self,
            X: pd.DataFrame,
            y: Optional[Union[pd.Series, np.ndarray]] = None
           ) -> Tuple[pd.DataFrame, Optional[Union[pd.Series, np.ndarray]]]:
        """
        Fit preprocessing pipeline: clean NaNs, feature elimination, PCA.

        Returns cleaned & processed X and y.
        """
        # Clean NaNs from X and y
        if y is not None:
            df = X.copy()
            df['__target__'] = y
            df_clean = df.dropna()
            X_clean = df_clean.drop(columns='__target__')
            y_clean = df_clean['__target__']
        else:
            X_clean = X.dropna()
            y_clean = None

        # Feature elimination
        if self.feature_elimination and y_clean is not None:
            X_clean = self._feature_elimination(X_clean, y_clean)
        else:
            self.selected_features = X_clean.columns.tolist()

        # PCA
        if self.pca:
            X_clean = self._apply_pca(X_clean)
        else:
            self.pca_model = None

        return X_clean, y_clean

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing to new data.
        """
        # Clean NaNs
        X_clean = X.dropna()

        # Select features
        if self.selected_features is not None:
            X_clean = X_clean[self.selected_features]

        # PCA transform
        if self.pca and self.pca_model is not None:
            X_pca = self.pca_model.transform(X_clean)
            columns = [f'pca_{i+1}' for i in range(X_pca.shape[1])]
            return pd.DataFrame(X_pca, columns=columns, index=X_clean.index)

        return X_clean

    def fit_transform(self,
                      X: pd.DataFrame,
                      y: Optional[Union[pd.Series, np.ndarray]] = None) -> Tuple[pd.DataFrame, Optional[Union[pd.Series, np.ndarray]]]:
        return self.fit(X, y)

    def _feature_elimination(self,
                             X: pd.DataFrame,
                             y: Union[pd.Series, np.ndarray]) -> pd.DataFrame:
        if self.fe_method == 'lasso':
            lasso = LassoCV(cv=5, random_state=42, n_jobs=-1).fit(X, y)
            selector = SelectFromModel(lasso, prefit=True, max_features=self.fe_n_features)
            mask = selector.get_support()
            features_selected = X.columns[mask]
            if self.fe_n_features is not None and len(features_selected) > self.fe_n_features:
                coef_abs = np.abs(lasso.coef_[mask])
                top_idx = np.argsort(coef_abs)[-self.fe_n_features:]
                features_selected = features_selected[top_idx]
            self.selected_features = list(features_selected)
            return X[self.selected_features]

        elif self.fe_method == 'tree':
            tree_model = RandomForestRegressor(random_state=42, n_jobs=-1)
            tree_model.fit(X, y)
            importances = tree_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            if self.fe_n_features is not None:
                indices = indices[:self.fe_n_features]
            features_selected = X.columns[indices]
            self.selected_features = list(features_selected)
            return X[self.selected_features]

        elif self.fe_method == 'correlation':
            corrs = X.apply(lambda col: np.abs(np.corrcoef(col, y)[0, 1]))
            corrs = corrs.fillna(0)
            selected = corrs.sort_values(ascending=False)
            if self.fe_n_features is not None:
                selected = selected.iloc[:self.fe_n_features]
            self.selected_features = list(selected.index)
            return X[self.selected_features]

        else:
            # No feature elimination
            self.selected_features = X.columns.tolist()
            return X

    def _apply_pca(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.pca_n_components is None or self.pca_n_components >= X.shape[1]:
            self.pca_model = None
            return X
        self.pca_model = PCA(n_components=self.pca_n_components, random_state=42)
        X_reduced = self.pca_model.fit_transform(X)
        columns = [f'pca_{i+1}' for i in range(X_reduced.shape[1])]
        return pd.DataFrame(X_reduced, columns=columns, index=X.index)