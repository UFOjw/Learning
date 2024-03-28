# Gradient-Boosting

Gradient boosting is a machine learning technique that consistently improves the quality of predictions by combining weak models into a strong model.

The main idea is to train a new model that will correct the "errors" of the previous model, gradually improving the quality of the predictions at each iteration.

This is achieved by finding the gradient of the loss function (eg MSE or MAE) with respect to the predictions of the current ensemble of models and training the new model on those errors.

![1](https://storage.yandexcloud.net/klms-public/production/learning-content/55/1255/22321/64360/300276/gradient_boosting_2.gif)

The implemented algorithm represents an ensemble of decision trees. The class uses the `DecisionTreeRegressor` from the `sklearn` library to build trees.

The class is initialized with the following arguments:
* `n_estimators` - number of trees in the ensemble
* `learning_rate` - coefficient taking into account the predictions of subsequent models
* `max_depth` - decision tree depth
* `min_samples_split` - the number of instances per leaf that allows the tree to continue building (stopping criterion)
* `loss` - loss function (default MSE. Function should return error and loss gradient)
* `verbose` - whether to display intermediate results
* `subsample_size` - amount of data used to train one tree (in fractions)
* `replace` - whether to sample with return

The class has a `fit` method for building a model, which takes two parameters `X_train`, `y_train` and the `predict` method for predicting new `X_test` data.
