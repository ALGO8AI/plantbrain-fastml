
# PlantBrain-FastML: Automated Machine Learning Framework

**PlantBrain-FastML** is a Python framework designed to accelerate the process of training, evaluating, and tuning machine learning models. It provides a high-level API to automate boilerplate code for model comparison, preprocessing, and hyperparameter optimization, allowing you to go from a dataset to a tuned model with just a few lines of code.

---

## 📚 Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start Examples](#quick-start-examples)
  - [Regression Example](#1-regression-example)
  - [Classification Example](#2-classification-example)
- [Detailed API Reference](#detailed-api-reference)
  - [RegressorManager & ClassifierManager](#regressormanager--classifiermanager)
  - [evaluate_all() - Core Method](#evaluate_all---core-method)
  - [get_best_model()](#get_best_model)
  - [get_hyperparameters()](#get_hyperparameters)
- [How to Contribute](#how-to-contribute)
- [License](#license)

---

##  Key Features

- **Automated Model Comparison**: Evaluate dozens of models for both regression and classification tasks simultaneously to find the best performer.
- **Integrated Preprocessing**: Seamlessly apply feature elimination (e.g., using Lasso) and scaling as part of the evaluation pipeline.
- **Powerful Hyperparameter Tuning**: Built-in support for Optuna to automatically find the best hyperparameters for your models.
- **Parallel Processing**: Speed up model evaluation significantly by utilizing all available CPU cores.
- **Rich Reporting**: Generates a comprehensive pandas DataFrame with cross-validation scores and test set metrics for easy analysis.
- **Extensible**: Easily add your own custom model wrappers to expand the framework.

---

##  Installation

To install the library, clone this repository and install it in editable mode using pip.

```bash
git clone https://github.com/YOUR_USERNAME/plantbrain-fastml.git
cd plantbrain-fastml
pip install -e .
```

---

##  Quick Start Examples

### 1️⃣ Regression Example

```python
import pandas as pd
from sklearn.datasets import load_diabetes
from plantbrain_fastml.managers.regressor_manager import RegressorManager

# Load Data
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.Series(diabetes.target, name='target')

# Initialize the Manager
reg_manager = RegressorManager()

# Evaluate All Models
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

# Get the Best Model
best_model_name, best_model_object = reg_manager.get_best_model(metric='rmse', higher_is_better=False)
all_hyperparams = reg_manager.get_hyperparameters()

print(f"--- Best Regressor: {best_model_name} ---")
print("Tuned Hyperparameters:")
print(all_hyperparams[best_model_name])
```

---

### 2️⃣ Classification Example

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from plantbrain_fastml.managers.classifier_manager import ClassifierManager

# Load Data
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target, name='target')

# Initialize the Manager
cls_manager = ClassifierManager()

# Evaluate All Models
results = cls_manager.evaluate_all(
    X,
    y,
    hypertune=True,
    hypertune_params={'n_trials': 30},
    n_jobs=-1
)

# Get the Best Model
best_model_name, best_model_object = cls_manager.get_best_model(metric='roc_auc', higher_is_better=True)
all_hyperparams = cls_manager.get_hyperparameters()

print(f"--- Best Classifier: {best_model_name} ---")
print("Tuned Hyperparameters:")
print(all_hyperparams[best_model_name])
```

---

##  Detailed API Reference

### RegressorManager & ClassifierManager

These are the main entry points to the library.

- `RegressorManager()` — for regression tasks
- `ClassifierManager()` — for classification tasks

### evaluate_all() - Core Method

This method runs the entire evaluation pipeline.

**Signature:**
```python
manager.evaluate_all(
    X, y, metrics=None, cv_folds=5, test_size=0.2,
    feature_elimination=False, fe_method=None, fe_n_features=None,
    hypertune=False, hypertune_params=None, hypertune_metrics=None,
    n_jobs=1
)
```

**Key Parameters:**

- `X`: (pd.DataFrame) input features (numeric)
- `y`: (pd.Series) target variable
- `metrics`: Optional custom metric dictionary
- `cv_folds`: Cross-validation folds
- `test_size`: Size of test split
- `feature_elimination`: Enable/disable feature selection
- `fe_method`: `'lasso'`, `'tree'`, or `'correlation'`
- `fe_n_features`: Number of features to select
- `hypertune`: Enable Optuna tuning
- `hypertune_params`: Dict like `{'n_trials': 50}`
- `hypertune_metrics`: Metric name for tuning
- `n_jobs`: Parallel jobs (`-1` = all cores)

**Returns:**  
`pd.DataFrame` with CV scores and test set metrics for each model.

---

### get_best_model()

Retrieve the best-performing model.

**Signature:**
```python
manager.get_best_model(metric: str, higher_is_better: bool = True)
```

**Returns:**
- `str`: Best model name
- `object`: Fitted model instance

---

### get_hyperparameters()

Get tuned hyperparameters after `evaluate_all(hypertune=True)`.

**Signature:**
```python
manager.get_hyperparameters()
```

**Returns:**  
`Dict[str, Dict]` — model names mapped to their best parameter dicts.

---

##  How to Contribute

Contributions are welcome! To contribute:

1. **Fork** the repository  
2. Create your branch:  
   ```bash
   git checkout -b feature/my-new-feature
   ```
3. Commit your changes:  
   ```bash
   git commit -am 'Add some feature'
   ```
4. Push to your branch:  
   ```bash
   git push origin feature/my-new-feature
   ```
5. Create a Pull Request

---

##  License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
