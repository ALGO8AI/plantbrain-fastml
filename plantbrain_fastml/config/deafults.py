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
