import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns the last-axis normalized array.
    """
    return beta + gamma * ((x - x.mean(axis = -1, keepdims = True))/np.sqrt(x.var(axis = -1, keepdims=True) + eps))