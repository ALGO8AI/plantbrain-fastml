from __future__ import annotations
import os, sys
import pandas as pd
import numpy as np
from typing import Any, Dict, Sequence, Union

from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.preprocessing import PowerTransformer
import joblib
import optuna

from plantbrain_fastml.utils.metrics import forecasting_metrics
from plantbrain_fastml.utils.preprocessing import default_preprocessor
from plantbrain_fastml.base.base_forecaster import BaseForecaster

class ArimaForecaster(BaseForecaster):
    """
    A univariate (no-exogenous) ARIMA forecaster that supports
    * automatic (p,d,q) selection
    * Yeo-Johnson power-transform (reversible)
    * Optuna hyper-parameter search via ``search_space``.
    """
    name: str = "ArimaForecaster"
    def __init__(
        self,
        forecast_horizon: int = 1,
        *,
        transformation: str | None = "power_transformer",
        arima_orders: tuple[int, int, int] | None = None,
        val_set: float = 0.2,
        **params,
    ):
        super().__init__(forecast_horizon=forecast_horizon, **params)

        self.val_set = val_set
        self.transformation = transformation
        self.arima_orders = arima_orders

        self._transformer: PowerTransformer | None = None
        self._y_index: pd.Index | None = None

    def preprocess_data(
        self,
        data: Union[pd.DataFrame, pd.Series],
        *,
        target: str | Sequence[str] = None,
        **_,
    ) -> tuple[pd.DataFrame | None, pd.Series]:
        """
        ARIMA is univariate; all we really need is the target series.
        """
        try:
            if isinstance(data, pd.Series):
                y = data
            else:  
                if target is None or target not in data.columns:
                    raise ValueError("`target` column must be supplied for DataFrame input.")
                y = data[target]

            return None, y.dropna()
        except Exception as exc:
            # failure_logger.error("ARIMA preprocessing failed.", exc_info=True)
            raise

    def train(self, X: pd.DataFrame | None, y: pd.Series) -> None:
        """Fit an ARIMA model to *y* (ignores *X*)."""
        try:
            self._y_index = y.index

            if self.transformation == "power_transformer":
                self._transformer = PowerTransformer(method="yeo-johnson")
                y_trans = self._transformer.fit_transform(y.values.reshape(-1, 1)).flatten()
                y_train = pd.Series(y_trans, index=y.index)
            else:
                y_train = y

            if self.arima_orders is None:
                auto = auto_arima(
                    y_train,
                    seasonal=False,
                    trace=False,
                    suppress_warnings=True,
                    stepwise=True,
                    error_action="ignore",
                )
                self.arima_orders = auto.order
                print(f"Auto ARIMA selected order: {self.arima_orders}")
                # forecast_logger.info(f"auto_arima selected order {self.arima_orders}")

            self.model = ARIMA(y_train, order=self.arima_orders).fit()
            # forecast_logger.info(f"Fitted ARIMA{self.arima_orders}")

        except Exception as exc:
            # failure_logger.error("ARIMA training failed.", exc_info=True)
            raise

    def predict(
        self,
        X: pd.DataFrame | None = None,
        *,
        steps: int | None = None,
    ) -> pd.Series:
        """
        * If *X* is supplied → in-sample forecasts of identical length.
        * Else → out-of-sample forecast for ``steps`` (default = horizon).
        """
        if self.model is None:
            raise RuntimeError("Call `train` before `predict`.")

        if X is not None:
            n = len(X)
            preds = self.model.predict(start=0, end=n - 1)

            idx = getattr(X, "index", None) or self._y_index[:n]
            preds = pd.Series(preds, index=idx)

        else:
            steps = steps or self.forecast_horizon
            preds = self.model.forecast(steps=steps)
            preds = pd.Series(preds)

        if self._transformer is not None:
            inv = self._transformer.inverse_transform(preds.values.reshape(-1, 1)).flatten()
            preds = pd.Series(inv, index=preds.index)

        return preds

    def search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Basic integer grid for p,d,q ∈ [0-5].  Feel free to enrich!
        """
        p = trial.suggest_int("p", 0, 20)
        d = trial.suggest_int("d", 0, 20)
        q = trial.suggest_int("q", 0, 20)
        return {"arima_orders": (p,d,q)}

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Nothing to save – train the model first.")
        joblib.dump(self.model, path)
        if self._transformer is not None:
            joblib.dump(self._transformer, path.replace(".pkl", "_transformer.pkl"))

    @classmethod
    def load(cls, path: str, **init_kw) -> "ArimaForecaster":
        """
        Reload a saved forecaster (model + transformer).
        """
        forecaster = cls(**init_kw)
        forecaster.model = joblib.load(path)
        transf_path = path.replace(".pkl", "_transformer.pkl")
        if joblib.os.path.exists(transf_path):
            forecaster._transformer = joblib.load(transf_path)
        # forecast_logger.info(f"Loaded ARIMA model from {path}")
        return forecaster