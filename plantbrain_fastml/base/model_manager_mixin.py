import pandas as pd
from typing import Dict, Any, Optional

class ModelManagerMixin:
    """
    Mixin class to manage multiple machine learning models with
    support for training, evaluation, and hyperparameter tuning.

    Features:
    - Add multiple models.
    - Train all models with optional feature elimination, PCA, and hypertuning.
    - Evaluate all models using cross-validation or train-test split.
    - Return results as a pandas DataFrame.
    - Retrieve best performing model based on a specified metric.
    """

    def __init__(self):
        """
        Initialize ModelManagerMixin with empty models and results.
        """
        self.models = {}
        self.results = pd.DataFrame()

    def add_model(self, name: str, model):
        """
        Add a model to the manager.

        Parameters:
        -----------
        name : str
            Unique name for the model.
        model : object
            Model instance implementing train, evaluate, hypertune methods.
        """
        self.models[name] = model

    def train_all(self,
                  X,
                  y,
                  feature_elimination: bool = False,
                  fe_method: Optional[str] = None,
                  fe_n_features: Optional[int] = None,
                  pca: bool = False,
                  pca_n_components: Optional[int] = None,
                  hypertune: bool = False,
                  hypertune_params: Optional[Dict[str, Any]] = None):
        """
        Train all models, optionally performing hypertuning.

        Parameters:
        -----------
        X : array-like or DataFrame
            Training features.
        y : array-like
            Target variable.
        feature_elimination : bool, default False
            Whether to perform feature elimination during training.
        fe_method : str or None, default None
            Feature elimination method ('lasso', 'tree', 'correlation').
        fe_n_features : int or None, default None
            Number of features to select during feature elimination.
        pca : bool, default False
            Whether to perform PCA dimensionality reduction.
        pca_n_components : int or None, default None
            Number of PCA components to keep.
        hypertune : bool, default False
            Whether to perform hyperparameter tuning before training.
        hypertune_params : dict or None, default None
            Additional parameters for hypertuning method.
        """
        hypertune_params = hypertune_params or {}

        for name, model in self.models.items():
            if hypertune:
                best_params = model.hypertune(X, y,
                                             feature_elimination=feature_elimination,
                                             fe_method=fe_method,
                                             fe_n_features=fe_n_features,
                                             pca=pca,
                                             pca_n_components=pca_n_components,
                                             **hypertune_params)
                print(f"[{name}] Best params: {best_params}")
            else:
                model.train(X, y,
                            feature_elimination=feature_elimination,
                            fe_method=fe_method,
                            fe_n_features=fe_n_features,
                            pca=pca,
                            pca_n_components=pca_n_components)

    def evaluate_all(self,
                    X,
                    y,
                    metrics: Optional[Dict[str, Any]] = None,
                    cv_folds: int = 5,
                    test_size: float = 0.2,
                    feature_elimination: bool = False,
                    fe_method: Optional[str] = None,
                    fe_n_features: Optional[int] = None,
                    pca: bool = False,
                    pca_n_components: Optional[int] = None,
                    hypertune: bool = False,
                    hypertune_params: Optional[Dict[str, Any]] = None,
                    return_hypertune_params: bool = False) -> pd.DataFrame:
        """
        Evaluate all models on given data, automatically calling train_all first.

        Returns a DataFrame with metrics only, and stores hypertune params
        in a separate dict attribute `self.hypertune_params` accessible by user.
        """
        hypertune_params = hypertune_params or {}

        self.train_all(X, y,
                    feature_elimination=feature_elimination,
                    fe_method=fe_method,
                    fe_n_features=fe_n_features,
                    pca=pca,
                    pca_n_components=pca_n_components,
                    hypertune=hypertune,
                    hypertune_params=hypertune_params)

        results_list = []
        self.hypertune_params = {}  # reset container for hypertune params

        for name, model in self.models.items():
            scores = model.evaluate(X, y,
                                    metrics=metrics,
                                    cv_folds=cv_folds,
                                    test_size=test_size,
                                    feature_elimination=feature_elimination,
                                    fe_method=fe_method,
                                    fe_n_features=fe_n_features,
                                    pca=pca,
                                    pca_n_components=pca_n_components,
                                    return_cv_scores=False)
            row = {'model': name}
            row.update(scores)
            results_list.append(row)

            if return_hypertune_params and hypertune:
                # Store hypertune params separately keyed by model name
                self.hypertune_params[name] = dict(model.params)  # copy to avoid mutation

        self.results = pd.DataFrame(results_list).set_index('model')
        return self.results


    def hypertune_all(self,
                     X,
                     y,
                     n_trials: int = 20,
                     timeout: Optional[int] = None,
                     metric: str = 'rmse',
                     direction: str = 'minimize',
                     feature_elimination: bool = False,
                     fe_method: Optional[str] = None,
                     fe_n_features: Optional[int] = None,
                     pca: bool = False,
                     pca_n_components: Optional[int] = None,
                     cv_folds: int = 5,
                     test_size: float = 0.2,
                     random_state: int = 42) -> None:
        """
        Perform hyperparameter tuning on all models.

        Parameters:
        -----------
        X : array-like or DataFrame
            Features data.
        y : array-like
            Target variable.
        n_trials : int, default 20
            Number of optimization trials.
        timeout : int or None, default None
            Timeout in seconds for optimization.
        metric : str, default 'rmse'
            Metric to optimize.
        direction : str, default 'minimize'
            Direction of optimization ('minimize' or 'maximize').
        feature_elimination : bool, default False
            Whether to apply feature elimination during tuning.
        fe_method : str or None, default None
            Feature elimination method.
        fe_n_features : int or None, default None
            Number of features for elimination.
        pca : bool, default False
            Whether to apply PCA during tuning.
        pca_n_components : int or None, default None
            Number of PCA components.
        cv_folds : int, default 5
            Number of cross-validation folds.
        test_size : float, default 0.2
            Test set size if not using CV.
        random_state : int, default 42
            Random seed.
        """
        for name, model in self.models.items():
            best_params = model.hypertune(X, y,
                                         n_trials=n_trials,
                                         timeout=timeout,
                                         metric=metric,
                                         direction=direction,
                                         feature_elimination=feature_elimination,
                                         fe_method=fe_method,
                                         fe_n_features=fe_n_features,
                                         pca=pca,
                                         pca_n_components=pca_n_components,
                                         cv_folds=cv_folds,
                                         test_size=test_size,
                                         random_state=random_state)
            print(f"[{name}] Best params: {best_params}")

    def get_best_model(self, metric: str, higher_is_better: bool = True):
        """
        Retrieve the best performing model according to specified metric.

        Parameters:
        -----------
        metric : str
            Metric name to compare models on.
        higher_is_better : bool, default True
            Whether higher metric values indicate better performance.

        Returns:
        --------
        tuple
            (best_model_name, best_model_instance)

        Raises:
        -------
        ValueError
            If evaluation results are empty or metric not found.
        """
        best_name = None
        best_score = None
        if self.results.empty:
            raise ValueError("No evaluation results available. Please run evaluate_all first.")

        for name, scores in self.results.iterrows():
            score = scores.get(metric)
            if score is None:
                continue
            if best_score is None or (score > best_score if higher_is_better else score < best_score):
                best_score = score
                best_name = name
        if best_name is None:
            raise ValueError(f"No model has metric '{metric}' in results.")
        return best_name, self.models[best_name]
