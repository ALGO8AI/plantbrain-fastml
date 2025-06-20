
# PlantBrain-FastML: Automated Machine Learning Framework

**PlantBrain-FastML** is a Python framework designed to accelerate the process of training, evaluating, and tuning machine learning models. It provides a high-level API to automate boilerplate code for model comparison, preprocessing, and hyperparameter optimization, allowing you to go from a dataset to a tuned model with just a few lines of code.

---

## Key Features

- **Automated Model Comparison**  
  Evaluate dozens of models for both regression and classification tasks simultaneously to find the best performer.

- **Integrated Preprocessing**  
  Seamlessly apply feature elimination (e.g., using Lasso) and scaling as part of the evaluation pipeline.

- **Powerful Hyperparameter Tuning**  
  Built-in support for Optuna to automatically find the best hyperparameters for your models.

- **Parallel Processing**  
  Speed up model evaluation significantly by utilizing all available CPU cores.

- **Rich Reporting**  
  Generates a comprehensive pandas `DataFrame` with cross-validation scores and test set metrics for easy analysis.

- **Extensible**  
  Easily add your own custom model wrappers to expand the framework.

---

## Installation

To install the library, clone this repository and install it in editable mode using `pip`.

```bash
git clone https://github.com/YOUR_USERNAME/plantbrain-fastml.git
cd plantbrain-fastml
pip install -e .
```

---

## Quick Start: Usage Examples

### 1️⃣ Regression Example  
Find the best regression model for the Diabetes dataset with automated hyperparameter tuning and feature elimination.

```python
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
    hypertune_params={'n_trials': 25},  # More trials for a better search
    n_jobs=-1,                          # Use all available CPU cores
    feature_elimination=True,
    fe_method='lasso',
    fe_n_features=5
)

# 4. Get the Best Model
best_model_name, best_model_object = reg_manager.get_best_model(metric='rmse', higher_is_better=False)
all_hyperparams = reg_manager.get_hyperparameters()

print(f"--- Best Regressor: {best_model_name} ---")
print("Tuned Hyperparameters:")
print(all_hyperparams[best_model_name])
```

---

### 2️⃣ Classification Example  
Find the best classifier for the Breast Cancer dataset, letting Optuna tune the loss functions and other parameters.

```python
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
best_model_name, best_model_object = cls_manager.get_best_model(metric='roc_auc', higher_is_better=True)
all_hyperparams = cls_manager.get_hyperparameters()

print(f"--- Best Classifier: {best_model_name} ---")
print("Tuned Hyperparameters:")
print(all_hyperparams[best_model_name])
```

---

## How to Contribute

Contributions are welcome! If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

1. Fork the repository.
2. Create your feature branch:  
   ```bash
   git checkout -b feature/my-new-feature
   ```
3. Commit your changes:  
   ```bash
   git commit -am 'Add some feature'
   ```
4. Push to the branch:  
   ```bash
   git push origin feature/my-new-feature
   ```
5. Create a new Pull Request.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
