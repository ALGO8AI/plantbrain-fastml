from plantbrain_fastml.models.regressors.linear_regression import LinearRegressionRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load California housing dataset as alternative
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=42)

model = LinearRegressionRegressor()
model.train(X_train, y_train)
print(model.evaluate(X_test, y_test))
