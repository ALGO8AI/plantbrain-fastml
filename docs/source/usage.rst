#############
Usage Guide
#############

This guide provides examples of the two primary use cases for ``plantbrain-fastml``: regression and classification.

Regression Example
------------------

Find the best regression model for the Diabetes dataset with automated hyperparameter tuning and feature elimination.

.. code-block:: python

   import pandas as pd
   from sklearn.datasets import load_diabetes
   from plantbrain_fastml.managers.regressor_manager import RegressorManager

   # 1. Load Data
   diabetes = load_diabetes()
   X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
   y = pd.Series(diabetes.target, name='target')

   # 2. Initialize the Manager
   reg_manager = RegressorManager()

   # 3. Evaluate All Models
   results = reg_manager.evaluate_all(
       X,
       y,
       hypertune=True,
       hypertune_params={'n_trials': 25},
       n_jobs=-1,
       feature_elimination=True,
       fe_method='lasso',
       fe_n_features=5
   )

   # 4. Get the Best Model
   best_model_name, _ = reg_manager.get_best_model(metric='rmse', higher_is_better=False)
   print(f"Best Regressor: {best_model_name}")


Classification Example
----------------------

Find the best classifier for the Breast Cancer dataset, letting Optuna tune the loss functions and other parameters.

.. code-block:: python

   import pandas as pd
   from sklearn.datasets import load_breast_cancer
   from plantbrain_fastml.managers.classifier_manager import ClassifierManager

   # 1. Load Data
   cancer = load_breast_cancer()
   X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
   y = pd.Series(cancer.target, name='target')

   # 2. Initialize the Manager
   cls_manager = ClassifierManager()

   # 3. Evaluate All Models
   results = cls_manager.evaluate_all(
       X,
       y,
       hypertune=True,
       hypertune_params={'n_trials': 30},
       n_jobs=-1
   )

   # 4. Get the Best Model
   best_model_name, _ = cls_manager.get_best_model(metric='roc_auc', higher_is_better=True)
   print(f"Best Classifier: {best_model_name}")
