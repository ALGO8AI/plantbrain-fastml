import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score,precision_score, recall_score,f1_score,roc_auc_score

# Regression metrics dictionary
regression_metrics = {
    "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    "mae": mean_absolute_error,
    "r2": r2_score,
}

# Classification metrics dictionary
classification_metrics = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
    "roc_auc":roc_auc_score,
}

# Forecasting metrics (reuse regression metrics)
forecasting_metrics = {
    "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    "mape": lambda y_true, y_pred: np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
    "mae": mean_absolute_error,
    "mse": lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2),
}
