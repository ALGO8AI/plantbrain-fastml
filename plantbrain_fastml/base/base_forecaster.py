from abc import ABC, abstractmethod
from typing import Any, Dict
import optuna
import pandas as pd
from plantbrain_fastml.utils.preprocessing import default_preprocessor
from plantbrain_fastml.utils.metrics import forecasting_metrics

class BaseForecaster(ABC):
    def __init__(self, forecast_horizon:int = 1, **params):
        self.params = params
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.preprocessor = default_preprocessor()

    @abstractmethod
    def preprocess_data(self, data: pd.DataFrame, **kwargs):
        """
        Preprocess data for training and forecasting.
        Must be implemented by derived classes.
        """
        pass

    @abstractmethod
    def train(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    def evaluate(self, X, y, metrics=None) -> Dict[str, float]:
        if metrics is None:
            metrics = forecasting_metrics
        y_pred = self.predict(X)
        results = {name: fn(y, y_pred) for name, fn in metrics.items()}
        return results
    
    def set_params(self, **params):
        self.params.update(params)
    
    def walk_forward_prediction(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series ):
        """
        Perform walk-forward prediction on the entire test dataset.

        Parameters:
        - X_train (pd.DataFrame): Initial training features.
        - y_train (pd.Series): Initial training target values.
        - X_test (pd.DataFrame): Test features.
        - y_test (pd.Series): Test target values.

        Returns:
        - pd.DataFrame: Predicted values for each walk-forward iteration.
        """
        predictions = []
        start_idx = 0

        while start_idx < len(X_test):
            end_idx = min(start_idx + self.forecast_horizon, len(X_test))
            model = self.train(X_train, y_train)

            forecast = self.predict(X_test=X_train, y_test=y_train)

            predictions.extend(forecast)

            X_test_slice = X_test.iloc[start_idx:end_idx]

            X_train = pd.concat([X_train, X_test_slice], axis=0)
            y_train = pd.concat([y_train, y_test.iloc[start_idx:end_idx]], axis=0)

            start_idx += self.forecast_horizon

        return pd.DataFrame(predictions, index=y_test.index[:len(predictions)], columns=["Prediction"])
    
    def hypertune(self, X, y, n_trials=20, timeout=None, metric='rmse', direction='minimize'):
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

    def get_params(self) -> dict:
        """
        Export model parameters for persistence.
        Returns a dict in the format { model_name: {param_key: value, ...} }.
        """
        params = {}
        for attr, val in self.__dict__.items():
            if attr.startswith('_') or attr in ['forecast_horizon']:
                params[attr] = val
        return {self.__class__.__name__: params}
