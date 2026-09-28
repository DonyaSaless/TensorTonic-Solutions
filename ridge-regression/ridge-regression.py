import numpy as np

def ridge_regression(X: list, y: list, lam: float) -> list:
    """
    Returns the ridge-regression weight vector.
    """
    # Write code here
    X = np.asarray(X, dtype = np.float64)
    y = np.asarray(y, dtype = np.float64)

    return np.linalg.inv(X.T@X + lam *np.identity(X.shape[1], dtype = np.float64)) @ X.T@y