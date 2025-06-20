API Reference
=============

This section provides the auto-generated API documentation for the core classes in ``plantbrain-fastml``.

Managers
--------

These are the high-level classes you will interact with directly.

.. autoclass:: plantbrain_fastml.managers.regressor_manager.RegressorManager
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: plantbrain_fastml.managers.classifier_manager.ClassifierManager
   :members:
   :undoc-members:
   :show-inheritance:

Base Classes
------------

These are the underlying abstract classes that provide the core logic. You can use them to extend the framework with new models.

.. autoclass:: plantbrain_fastml.base.model_manager_mixin.ModelManagerMixin
   :members: evaluate_all, get_best_model, get_hyperparameters
   :undoc-members:

.. autoclass:: plantbrain_fastml.base.base_regressor.BaseRegressor
   :members: evaluate, search_space
   :undoc-members:

.. autoclass:: plantbrain_fastml.base.base_classifier.BaseClassifier
   :members: evaluate, search_space
   :undoc-members:

