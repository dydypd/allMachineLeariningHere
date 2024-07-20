from numba import njit
from numba import float64
import numpy as np


@njit(cache=True, parallel=True)
def fit_numba(X, y, n_iters=1000, lr=0.01) -> float64:
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    for _ in range(n_iters):
        y_predicted = np.dot(X, weights) + bias
        dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
        db = (1 / n_samples) * np.sum(y_predicted - y)

        weights -= lr * dw
        bias -= lr * db

    return weights, bias


@njit()
def predict_numba(X, weights, bias) -> float64:
    return np.dot(X, weights) + bias
