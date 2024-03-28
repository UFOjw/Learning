from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class Node:
    """Decision tree node."""
    feature: int = None
    threshold: float = None
    n_samples: int = None
    value: int = None
    mse: float = None
    left: Node = None
    right: Node = None

@dataclass
class DecisionTreeRegressor:
    """Decision tree regressor."""
    max_depth: int
    min_samples_split: int = 2

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeRegressor:
        """Build a decision tree regressor from the training set (X, y)."""
        self.n_features_ = X.shape[1]
        self.tree_ = self._split_node(X, y)
        return self

    def _mse(self, y: np.ndarray) -> float:
        """Compute the mse criterion for a given set of target values."""
        target = np.mean(y)
        count_of_preds = y.size
        return np.sum(np.power(y - target, 2)) / count_of_preds

    def _weighted_mse(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """Compute the weithed mse criterion for a two given sets of target values"""
        target_left = np.mean(y_left)
        target_right = np.mean(y_right)
        count_of_preds_left = y_left.size
        count_of_preds_right = y_right.size
        l2_left = np.sum(np.power(y_left - target_left, 2))
        l2_right = np.sum(np.power(y_right - target_right, 2))
        return (l2_left + l2_right) / (count_of_preds_left + count_of_preds_right)

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
        """Find the best split for a node."""
        best_thr = None
        best_wmse = None
        best_idx = None
        num_of_features = X[0].size
        for idx in range(num_of_features):
            feature = X[:, idx]
            for val in feature:
                mask = feature <= val
                wmse = self._weighted_mse(y[mask], y[~mask])
                if best_wmse is None or best_wmse > wmse:
                    best_wmse = wmse
                    best_thr = val
                    best_idx = idx
        return best_idx, best_thr

    def _split_node(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Split a node and return the resulting left and right child nodes."""
        n_samples_ = y.size
        value_ = round(np.mean(y))
        mse_ = self._mse(y)
        node = Node(n_samples = n_samples_, value = value_, mse = mse_)

        if (self.max_depth is not None and depth == self.max_depth) or n_samples_ <= self.min_samples_split:
            return node
        node.feature, node.threshold = self._best_split(X, y)
        mask = X[:, node.feature] <= node.threshold
        node.left = self._split_node(X[mask], y[mask], depth + 1)
        node.right = self._split_node(X[~mask], y[~mask], depth + 1)
        return node
    
    def as_json(self) -> str:
        """Return the decision tree as a JSON string."""
        file = str(self._as_json(self.tree_))
        
        return file.replace("'", "\"")

    def _as_json(self, node: Node) -> Dict:
        """Return the decision tree as a JSON string. Execute recursively."""
        if node.left is None:
            text = {
                    "value": node.value,
                    "n_samples": node.n_samples,
                    "mse": round(node.mse, 2)
                    }
        else:
            text = {
                    "feature": node.feature,
                    "threshold": node.threshold,
                    "n_samples": node.n_samples,
                    "mse": round(node.mse, 2),
                    "left": self._as_json(node.left),
                    "right": self._as_json(node.right)
                    }
        return text
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regression target for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array of shape (n_samples,)
            The predicted values.
        """
        y = np.zeros(X.shape[0])
        for idx, row in enumerate(X):
            y[idx] = self._predict_one_sample(row)
        return y


    def _predict_one_sample(self, features: np.ndarray) -> int:
        """Predict the target value of a single sample."""
        tree = self.tree_
        while tree.left is not None:
            feature = tree.feature
            threshold = tree.threshold
            if features[feature] <= threshold:
                tree = tree.left
            else:
                tree = tree.right
        return tree.value
