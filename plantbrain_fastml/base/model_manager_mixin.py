# plantbrain_fastml/base/model_manager_mixin.py
import pandas as pd
from typing import Dict, Any, Optional
import optuna
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from plantbrain_fastml.utils.helpers import get_effective_n_jobs


def _eval_single_model(name: str,
                       model,
                       X,
                       y,
                       metrics,
                       cv_folds,
                       test_size,
                       test_split_type,
                       feature_elimination,
                       fe_method,
                       fe_n_features,
                       pca,
                       pca_n_components,
                       return_plots,
                       hypertune,
                       hypertune_params,hypertune_metrics):
    """
    Evaluate a single model on given data with options.

    Runs model.evaluate(...) and returns results.
    """
    eval_result = model.evaluate(
        X, y,
        metrics=metrics,
        cv_folds=cv_folds,
        test_size=test_size,
        test_split_type=test_split_type,
        feature_elimination=feature_elimination,
        fe_method=fe_method,
        fe_n_features=fe_n_features,
        pca=pca,
        pca_n_components=pca_n_components,
        return_plots=return_plots,
        hypertune=hypertune,
        hypertune_params=hypertune_params,
        hypertune_metrics=hypertune_metrics
    )
    return name, eval_result


class ModelManagerMixin:
    """
    Manage multiple ML models with simple train_all and advanced evaluate_all.

    - train_all: just calls model.train (no hypertuning)
    - evaluate_all: calls model.evaluate (includes preprocessing, hypertuning, CV, test split)
    - Supports parallel evaluation with n_jobs using ProcessPoolExecutor for CPU-bound tasks
    """

    def __init__(self):
        self.models = {}
        self.results = pd.DataFrame()
        self.cv_results = {}
        self.test_results = {}

    def add_model(self, name: str, model):
        """Add a model instance with unique name."""
        self.models[name] = model

    def train_all(self, X, y):
        """Simple training on all models, no hypertuning or preprocessing."""
        for name, model in self.models.items():
            model.train(X, y)


    def evaluate_all(self,
                     X,
                     y,
                     metrics: Optional[Dict[str, Any]] = None,
                     cv_folds: int = 5,
                     test_size: float = 0.2,
                     test_split_type: str = 'random',
                     feature_elimination: bool = False,
                     fe_method: Optional[str] = None,
                     fe_n_features: Optional[int] = None,
                     pca: bool = False,
                     pca_n_components: Optional[int] = None,
                     hypertune: bool = False,
                     hypertune_params: Optional[Dict[str, Any]] = None,
                     hypertune_metrics: str = "rmse", #self.hypertune_metrics
                     return_plots: bool = True,
                     n_jobs: int = 1) -> pd.DataFrame:
        """
        Evaluate all models via their .evaluate() method, optionally in parallel.

        Parameters:
        -----------
        n_jobs : int
            Number of parallel workers (1 means no parallelism).

        Returns:
        --------
        pd.DataFrame of results indexed by model name.
        """

        hypertune_params = hypertune_params or {}

        # Suppress Optuna logs for clean output
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        logging.getLogger("optuna").setLevel(logging.WARNING)

        results_list = []
        self.cv_results = {}
        self.test_results = {}
        n_jobs=get_effective_n_jobs(n_jobs)
        if n_jobs == 1:
            # Sequential evaluation
            for name, model in self.models.items():
                name, eval_result = _eval_single_model(
                    name,
                    model,
                    X,
                    y,
                    metrics,
                    cv_folds,
                    test_size,
                    test_split_type,
                    feature_elimination,
                    fe_method,
                    fe_n_features,
                    pca,
                    pca_n_components,
                    return_plots,
                    hypertune,
                    hypertune_params,
                    hypertune_metrics
                )
                results_list.append((name, eval_result))
        else:
            # Parallel evaluation using ProcessPoolExecutor for CPU-bound work
            func = partial(
                _eval_single_model,
                X=X,
                y=y,
                metrics=metrics,
                cv_folds=cv_folds,
                test_size=test_size,
                test_split_type=test_split_type,
                feature_elimination=feature_elimination,
                fe_method=fe_method,
                fe_n_features=fe_n_features,
                pca=pca,
                pca_n_components=pca_n_components,
                return_plots=return_plots,
                hypertune=hypertune,
                hypertune_params=hypertune_params,
                hypertune_metrics=hypertune_metrics
            )

            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = {executor.submit(func, name, model): name for name, model in self.models.items()}
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        _, eval_result = future.result()
                        results_list.append((name, eval_result))
                    except Exception as e:
                        logging.error(f"Error evaluating model {name}: {e}")

        rows = []
        for name, eval_result in results_list:
            self.cv_results[name] = eval_result.get('cv_scores', {})
            self.test_results[name] = eval_result.get('test_scores', {})
            tuned_params=eval_result.get('tuned_params')
            if tuned_params:
                self.models[name].set_params(**tuned_params)

            row = {'model': name}
            for metric, (mean, std) in eval_result.get('cv_scores', {}).items():
                row[f'cv_{metric}_mean'] = mean
                row[f'cv_{metric}_std'] = std
            for metric, score in eval_result.get('test_scores', {}).items():
                row[f'test_{metric}'] = score

            rows.append(row)

        self.results = pd.DataFrame(rows).set_index('model')

        return self.results

    def get_best_model(self, metric: str, higher_is_better: bool = True):
        """
        Return best model by given metric, prefers test metric over CV mean.
        """
        if self.results.empty:
            raise ValueError("No evaluation results available. Run evaluate_all first.")

        best_name = None
        best_score = None

        for name, scores in self.results.iterrows():
            score = scores.get(f'test_{metric}', None) or scores.get(f'cv_{metric}_mean', None)
            if score is None:
                continue
            if best_score is None or (score > best_score if higher_is_better else score < best_score):
                best_score = score
                best_name = name

        if best_name is None:
            raise ValueError(f"No model found with metric '{metric}'.")

        return best_name, self.models[best_name]

    def get_hyperparameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Return stored hyperparameters dictionary keyed by model name.
        """
        hyperparams = {}
        for name, model in self.models.items():
            hyperparams[name] = getattr(model, 'params', {})
        return hyperparams

    def get_plots(self) -> Dict[str, Dict[str, Any]]:
        """
        Return stored plots dictionary keyed by model name.
        """
        plots = {}
        for name, model in self.models.items():
            plots[name] = getattr(model, 'latest_plots', {})
        return plots
