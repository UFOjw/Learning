# Gradient-Boosting

![1](https://storage.yandexcloud.net/klms-public/production/learning-content/55/1255/22321/64360/300276/gradient_boosting_2.gif)

# Gradient Boosting Regressor (NumPy + scikit-learn)

This project implements a custom `GradientBoostingRegressor` class using NumPy and `DecisionTreeRegressor` from `scikit-learn`. It demonstrates the core principles of gradient boosting for regression tasks, including loss computation, gradient descent, and decision tree fitting.

## Features

* Supports Mean Squared Error (MSE) loss
* Gradient computation
* Subsampling (stochastic boosting)
* Scikit-learn-compatible `fit()` and `predict()` methods
* Verbose mode for monitoring training

## File Overview

* `GradientBoostingRegressor` – main class implementing the boosting model
* `_mse()` – computes MSE and its gradient
* `_subsample()` – draws subsamples with or without replacement
* `fit()` – trains the model using boosting iterations
* `predict()` – performs prediction on new data

## Example Usage

```python
from GradientBoosting import GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

X, y = make_regression(n_samples=1000, n_features=20, noise=0.1)

model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, verbose=True)
model.fit(X, y)
predictions = model.predict(X)

print("MSE:", mean_squared_error(y, predictions))
```

## Requirements

* Python 3.x
* NumPy
* scikit-learn

```bash
pip install numpy scikit-learn
```
