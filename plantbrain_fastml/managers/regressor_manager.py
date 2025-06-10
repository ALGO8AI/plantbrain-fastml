# plantbrain_fastml/models/regressor.py

from plantbrain_fastml.models.regressors.linear_regression import LinearRegressionRegressor
from plantbrain_fastml.models.regressors.random_forest import RandomForestRegressorWrapper
from plantbrain_fastml.models.regressors.decesion_tree import DecisionTreeRegressorWrapper
from plantbrain_fastml.models.regressors.SVRegression import SVRWrapper
from plantbrain_fastml.models.regressors.KNN_regression import KNeighborsRegressorWrapper
from plantbrain_fastml.models.regressors.GradientBoostingRegression import GradientBoostingRegressorWrapper
from plantbrain_fastml.models.regressors.elastic_net import ElasticNet
from plantbrain_fastml.models.regressors.bayesian_ridge import BayesianRidge
from plantbrain_fastml.models.regressors.adaboost import AdaBoost
from plantbrain_fastml.models.regressors.lasso import Lasso
from plantbrain_fastml.models.regressors.ridge import Ridge

from plantbrain_fastml.base.model_manager_mixin import ModelManagerMixin

class RegressorManager(ModelManagerMixin):
    def __init__(self):
        super().__init__()
        # Register default models
        self.add_model("linear_regression", LinearRegressionRegressor())
        self.add_model("random_forest", RandomForestRegressorWrapper())
        self.add_model("decision_tree", DecisionTreeRegressorWrapper())
        self.add_model("svr", SVRWrapper())
        self.add_model("knn_regression", KNeighborsRegressorWrapper())
        self.add_model("gradient_boosting", GradientBoostingRegressorWrapper())
        self.add_model("elastic_net", ElasticNet())
        self.add_model("bayesian_ridge", BayesianRidge())
        self.add_model("adaboost", AdaBoost())
        self.add_model("lasso", Lasso())
        self.add_model("ridge", Ridge())

