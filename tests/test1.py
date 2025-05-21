from plantbrain_fastml.models.regressors.linear_regression import LinearRegressionRegressor
from plantbrain_fastml.managers.regressor_manager import RegressorManager
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load California housing dataset
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=42)

# Initialize manager and add model
manager = RegressorManager()
# model = LinearRegressionRegressor()
# manager.add_model('LinearRegression', model)

# Evaluate all with hypertuning enabled
results = manager.evaluate_all(
    X_train, y_train,
    hypertune=True,
    hypertune_params={'n_trials': 20},
    cv_folds=5,  # 5-fold cross-validation
    return_hypertune_params=True
)

print(results)
