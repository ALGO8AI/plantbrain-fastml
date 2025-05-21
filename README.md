# plantbrain-fastml

**An AutoML package by plantBrain for fast prototyping and experimentation in classification, regression, and forecasting tasks.**

---

## Features

- Unified base classes for regressors, classifiers, and forecasters  
- Built-in Optuna-powered hyperparameter tuning  
- Plug-and-play model architecture with popular algorithms  
- Model managers to train and compare multiple models easily  
- Out-of-the-box preprocessing and evaluation metrics  
- Scalable, maintainable, and extendable codebase  

---

## Installation

Install the package directly from PyPI:

```bash
pip install plantbrain-fastml
<!-- Package version: 0.1.0 -->
```
plantbrain-fastml requires Python 3.13.0 or higher and depends on the following packages:

scikit-learn (version 1.6.1 or newer)

pandas (version 2.2.3 or newer)

numpy (version 2.1.3 or newer)

optuna (version 4.1.0 or newer)

These dependencies will be installed automatically.

Alternatively, clone the repository and install in editable mode:

```bash

git clone https://github.com/ALGO8AI/plantbrain-fastml.git
cd plantbrain-fastml
pip install -e .
