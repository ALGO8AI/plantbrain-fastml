# Default hyperparameters for models

default_regressor_params = {
    "linear_regression": {
        "fit_intercept": True,
        "normalize": False,
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    },
    "decision_tree": {
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    }
}

default_classifier_params = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    }
}

default_forecaster_params = {
    "linear_regression": {
        "fit_intercept": True,
        "normalize": False,
    }
}

## Time series forecasting model parameters
default_exp_sm_params = {
            "trend": "add",
            "damped_trend": False,
            "seasonal": "add",
            "seasonal_periods": 12,
        }

default_lightgbm_params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.05,
            'num_leaves': 15,
            'max_depth': 4,
            'n_estimators': 100,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'max_bin': 127,
            'early_stopping_round': 10,
            'verbose': -1,
        }

default_prophet_params = {
            "growth": "linear",  
            "seasonality_prior_scale": 10.0,
            "holidays": None,  
            "yearly_seasonality": True,
            "weekly_seasonality": False,
            "daily_seasonality": False
        }

default_xgb_params = {
            "objective": "reg:squarederror",
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 200,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
        }