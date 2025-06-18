# plantbrain-fastml/plantbrain_fastml/managers/classifier_manager.py

from ..base.model_manager_mixin import ModelManagerMixin
from ..models.classifiers.random_forest_classifier import RandomForestClassifierWrapper
from ..models.classifiers.logistic_regression import LogisticRegressionWrapper
from ..models.classifiers.svc import SVCWrapper

class ClassifierManager(ModelManagerMixin):
    def __init__(self):
        super().__init__()
        # Store the classes, not instances
        self.add_model("random_forest", RandomForestClassifierWrapper())
        self.add_model("logistic_regression", LogisticRegressionWrapper())
        self.add_model("svc", SVCWrapper())

    def get_hypertune_metrics(self):
        return "roc_auc"