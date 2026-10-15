import numpy as np

def lasso_regression(X: list, y: list, alpha: float, lr: float, epochs: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X = np.asarray(X, dtype = np.float64)
    y = np.asarray(y, dtype = np.float64)
    bias = 0.0
    weights = np.zeros(X.shape[1], dtype = np.float64)
    
    for _ in range(epochs):
        err = X@weights + bias - y
        g_w = alpha * np.sign(weights)  + (2.0/len(y)) * X.T@err
        g_b = (2.0/len(y)) * sum(err)
        weights -= lr * g_w
        bias -= lr * g_b
    return [round(float(value), 4) for value in weights], round(float(bias), 4)
