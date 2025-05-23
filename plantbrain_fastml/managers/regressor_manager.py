# plantbrain_fastml/models/regressor.py

from plantbrain_fastml.models.regressors.linear_regression import LinearRegressionRegressor
from plantbrain_fastml.models.regressors.random_forest import RandomForestRegressorWrapper
from plantbrain_fastml.models.regressors.decesion_tree import DecisionTreeRegressorWrapper
from plantbrain_fastml.base.model_manager_mixin import ModelManagerMixin

class RegressorManager(ModelManagerMixin):
    def __init__(self):
        super().__init__()
        # Register default models
        self.add_model("linear_regression", LinearRegressionRegressor())
        self.add_model("random_forest", RandomForestRegressorWrapper())
        # self.add_model("decision_tree", DecisionTreeRegressorWrapper())
