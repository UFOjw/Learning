import numpy as np

from sklearn.tree import DecisionTreeRegressor


class GradientBoostingRegressor:
    """Gradient Boosting Regressor."""

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        loss="mse",
        verbose=False,
        replace=False,
        subsample_size=0.5
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss = loss
        self.verbose = verbose
        self.subsample_size = subsample_size,
        self.replace = replace

    def _mse(self, y_true, y_pred):
        """Mean squared error loss function and gradient."""
        grad = y_pred - y_true
        deviation = np.power(grad, 2)
        error = np.sum(deviation) / y_true.size
        return error, grad
    
    def _subsample(self, X, y):
        mask = np.random.choice(y.size, int(self.subsample_size[0] * y.size), replace=self.replace)
        sub_X = X[mask]
        sub_y = y[mask]
        return sub_X, sub_y

    def fit(self, X, y):
        """
        Fit the model to the data.

        Parameters
        ----------
        X :
            array-like of shape (n_samples, n_features)
        y :
            array-like of shape (n_samples,)

        Returns
        -------
        GradientBoostingRegressor
            The fitted model.

        """
        self.base_pred_ = np.mean(y)
        self.trees_ = []
        result = np.full(y.size, self.base_pred_)
        error, grad = None, None
        for _ in range(self.n_estimators):
            if self.loss == 'mse':
                error, grad = self._mse(y, result)
            else:
                error, grad = self.loss(y, result)
            regressor = DecisionTreeRegressor(
                                max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split)
            regressor.fit(X, grad * (-1))
            self.trees_.append(regressor)
            result += self.learning_rate * regressor.predict(X)
            if self.verbose:
                print(error)

    def predict(self, X):
        """
        Predict the target of new data.

        Parameters
        ----------
        X :
            array-like of shape (n_samples, n_features)

        Returns
        -------
        y :
            array-like of shape (n_samples,)
            The predict values.

        """
        predictions = self.base_pred_
        for tree in self.trees_:
            predictions += self.learning_rate * tree.predict(X)
        return predictions
