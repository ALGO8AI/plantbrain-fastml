# plantbrain-fastml/plantbrain_fastml/base/base_classifier.py

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import optuna
from .base_preprocessor import BasePreprocessor
from ..utils.metrics import classification_metrics

class BaseClassifier(ABC):
    def __init__(self, **params):
        self.params = params
        self.model = None
        self.preprocessor = BasePreprocessor()
        self.latest_plots = {}
        self.class_report = None

    def set_params(self, **params):
        self.params.update(params)
        if self.model is not None:
            self.model.set_params(**params)

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series):
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        pass

    def get_classification_report(self):
        return self.class_report
    

    def evaluate(self,
                 X: pd.DataFrame,
                 y: pd.Series,
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
                 hypertune_metrics: str = 'roc_auc',
                 return_plots: bool = True,
                 random_state: int = 42
                 ) -> Dict[str, Any]:
        
        if metrics is None:
            metrics = classification_metrics

        hypertune_params = hypertune_params or {}

        # Step 1: Split data - Added stratify=y for balanced splits in classification
        if test_split_type == 'random':
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=True, random_state=random_state, stratify=y)
        else:
            # Note: Stratified split is not possible with 'ordered' split
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Step 2: Initialize preprocessor with all arguments
        self.preprocessor = BasePreprocessor(
            feature_elimination=feature_elimination,
            fe_method=fe_method,
            fe_n_features=fe_n_features,
            pca=pca,
            pca_n_components=pca_n_components)
        
        X_train_proc, y_train_proc = self.preprocessor.fit_transform(X_train, y_train)

        # Step 3: Optional hypertuning
        if hypertune:
            if hypertune_metrics not in metrics:
                raise ValueError(f"Objective metric '{hypertune_metrics}' not found in metrics.")
            
            objective_metric_fn = metrics[hypertune_metrics]
            direction = 'maximize'

            def objective(trial: optuna.Trial) -> float:
                trial_params = self.search_space(trial)
                self.set_params(**trial_params)
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                cv_scores = []
                for train_idx, val_idx in cv.split(X_train_proc):
                    X_cv_train, X_cv_val = X_train_proc.iloc[train_idx], X_train_proc.iloc[val_idx]
                    y_cv_train, y_cv_val = y_train_proc.iloc[train_idx], y_train_proc.iloc[val_idx]
                    self.train(X_cv_train, y_cv_train)
                    
                    # Try to use predict_proba for relevant metrics, fall back to predict
                    try:
                        y_pred_proba_val = self.predict_proba(X_cv_val)[:, 1]
                        score = objective_metric_fn(y_cv_val, y_pred_proba_val)
                    except (AttributeError, IndexError):
                        y_pred_val = self.predict(X_cv_val)
                        score = objective_metric_fn(y_cv_val, y_pred_val)
                        
                    cv_scores.append(score)
                return np.mean(cv_scores)

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction=direction)
            study.optimize(objective, n_trials=hypertune_params.get('n_trials', 20))
            final_params = self.search_space(study.best_trial)
            self.set_params(**final_params)
            tuned_params = final_params

        # Step 4: Final CV on full training data with best params
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_scores = {name: [] for name in metrics.keys()}
        for train_idx, val_idx in cv.split(X_train_proc):
            # ... (CV logic remains the same)
            X_cv_train, X_cv_val = X_train_proc.iloc[train_idx], X_train_proc.iloc[val_idx]
            y_cv_train, y_cv_val = y_train_proc.iloc[train_idx], y_train_proc.iloc[val_idx]
            self.train(X_cv_train, y_cv_train)
            
            y_pred_val = self.predict(X_cv_val)
            y_pred_proba_val = self.predict_proba(X_cv_val)[:, 1]

            for name, metric_fn in metrics.items():
                if name in ['roc_auc', 'average_precision']:
                    score = metric_fn(y_cv_val, y_pred_proba_val)
                else:
                    score = metric_fn(y_cv_val, y_pred_val)
                cv_scores[name].append(score)

        cv_scores_summary = {name: (np.mean(scores), np.std(scores)) for name, scores in cv_scores.items()}
        
        # Step 5: Evaluate on the test set
        X_test_proc = self.preprocessor.transform(X_test)
        self.train(X_train_proc, y_train_proc) # Retrain on full processed training data
        y_pred_test = self.predict(X_test_proc)
        y_pred_proba_test = self.predict_proba(X_test_proc)[:, 1]

        test_scores = {}
        for name, fn in metrics.items():
            if name in ['roc_auc', 'average_precision']:
                test_scores[name] = fn(y_test, y_pred_proba_test)
            else:
                test_scores[name] = fn(y_test, y_pred_test)

        self.class_report = classification_report(y_test, y_pred_test, output_dict=True)
        # Note: Plots are not implemented in this version but can be added
        self.latest_plots = {}
        

        return {'cv_scores': cv_scores_summary, 'test_scores': test_scores, 'plots': self.latest_plots,'tuned_params':tuned_params}