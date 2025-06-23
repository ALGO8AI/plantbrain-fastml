from __future__ import annotations
import os
import joblib
from typing import Any, Dict, Sequence, Union

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import PowerTransformer
import optuna

from plantbrain_fastml.base.base_forecaster import BaseForecaster

class ExpSmoothingForecaster(BaseForecaster):
    """
    A univariate Exponential-Smoothing forecaster supporting
    * additive / multiplicative trend & seasonality
    * optional damping
    * automatic seasonality deactivation on short series
    * Yeo-Johnson power transform (reversible)
    * Optuna hyper-parameter optimisation via ``search_space``.
    """

    name: str = "ExpSmoothingForecaster"

    def __init__(
        self,
        forecast_horizon: int = 1,
        *,
        transformation: str | None = "power_transformer",
        exp_sm_params: Dict[str, Any] | None = None,
        val_set: float = 0.2,
        **base_kwargs,
    ):
        """
        Parameters
        ----------
        forecast_horizon : int
            Number of future steps to predict when ``predict()`` is called
            without *X* / *steps*.
        transformation : {"power_transformer", None}
            If set, a Yeo-Johnson power transform is fitted and inverted on prediction.
        exp_sm_params : dict or None
            Explicit parameter grid, e.g.
            ``dict(trend="add", damped_trend=False, seasonal="add", seasonal_periods=12)``.
            If *None*, defaults are used *or* values supplied by Optuna via ``search_space``.
        val_set : float
            Reserved fraction from the tail of the series for validation (unused here but
            kept for symmetry with other forecasters).
        **base_kwargs
            Passed to ``BaseForecaster`` (e.g. ``verbosity``).
        """
        super().__init__(forecast_horizon=forecast_horizon, **base_kwargs)

        self.transformation = transformation
        self.exp_sm_params = exp_sm_params or {
            "trend": "add",
            "damped_trend": False,
            "seasonal": "add",
            "seasonal_periods": 12,
        }
        self.val_set = val_set

        self._transformer: PowerTransformer | None = None
        self._y_index: pd.Index | None = None
        self.model: Any | None = None  

    def preprocess_data(
        self,
        data: Union[pd.DataFrame, pd.Series],
        *,
        target: str | Sequence[str] = None,
        **_,
    ) -> tuple[None, pd.Series]:
        """
        Keep only the univariate target series; drop NA rows.
        """
        if isinstance(data, pd.Series):
            y = data.dropna()
        else:
            if target is None or target not in data.columns:
                raise ValueError("`target` must be provided when passing a DataFrame.")
            y = data[target].dropna()

        return None, y  

    def train(self, X: None, y: pd.Series) -> None:  
        self._y_index = y.index

        if self.transformation == "power_transformer":
            self._transformer = PowerTransformer(method="yeo-johnson", standardize=True)
            y_train = pd.Series(
                self._transformer.fit_transform(y.values.reshape(-1, 1)).flatten(),
                index=y.index,
            )
        else:
            y_train = y

        p = self.exp_sm_params.copy()
        if p["seasonal"] is not None and len(y_train) < (p["seasonal_periods"] or 0):
            p["seasonal"] = None
            p["seasonal_periods"] = None

        model = ExponentialSmoothing(
            y_train,
            trend=p["trend"],
            damped_trend=p["damped_trend"],
            seasonal=p["seasonal"],
            seasonal_periods=p["seasonal_periods"],
            initialization_method="estimated",
        )
        self.model = model.fit()

    def predict(
        self,
        X: None = None,
        *,
        steps: int | None = None,
    ) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Call `train` before `predict`.")

        if X is not None:
            n = len(X)
            preds = self.model.predict(start=0, end=n - 1)
            idx = getattr(X, "index", None) or self._y_index[:n]
            preds = pd.Series(preds, index=idx)
        else:
            steps = steps or self.forecast_horizon
            preds = pd.Series(self.model.forecast(steps))

        if self._transformer is not None:
            inv = self._transformer.inverse_transform(preds.values.reshape(-1, 1)).flatten()
            preds = pd.Series(inv, index=preds.index)

        return preds

    def search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Example grid for Optuna hyper-parameter optimisation.
        Adjust ranges according to your domain & data frequency.
        """
        trend = trial.suggest_categorical("trend", [None, "add", "mul"])
        damped = trial.suggest_categorical("damped_trend", [False, True])
        seasonal = trial.suggest_categorical("seasonal", [None, "add", "mul"])
        seasons = trial.suggest_int("seasonal_periods", 4, 24, step=4) if seasonal else None

        return {
            "exp_sm_params": dict(
                trend=trend,
                damped_trend=damped,
                seasonal=seasonal,
                seasonal_periods=seasons,
            )
        }

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Nothing to save – train the model first.")
        joblib.dump(self.model, path)
        if self._transformer is not None:
            joblib.dump(self._transformer, path.replace(".pkl", "_transformer.pkl"))

    @classmethod
    def load(cls, path: str, **init_kw) -> "ExpSmoothingForecaster":
        """
        Reload a saved forecaster (model + transformer).
        """
        forecaster = cls(**init_kw)
        forecaster.model = joblib.load(path)
        transf_path = path.replace(".pkl", "_transformer.pkl")
        if os.path.exists(transf_path):
            forecaster._transformer = joblib.load(transf_path)
        return forecaster
