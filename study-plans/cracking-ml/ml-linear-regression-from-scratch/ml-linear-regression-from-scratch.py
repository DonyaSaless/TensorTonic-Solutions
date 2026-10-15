import numpy as np

def linear_regression_from_scratch(X: list, y: list, lr: float, epochs: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X = np.asarray(X, dtype = np.float64)
    y = np.asarray(y, dtype = np.float64)
    bias = 0.0
    weights = np.zeros(X.shape[1], dtype = np.float64)
    
    for _ in range(epochs):
        err = X@weights + bias -y
        gdw = 2.0*X.T@err
        gdb = 2.0*sum(err)
        weights -= (lr/len(y)) * gdw
        bias -= (lr/len(y)) * gdb
    return [round(float(value), 4) for value in weights], round(float(bias), 4)