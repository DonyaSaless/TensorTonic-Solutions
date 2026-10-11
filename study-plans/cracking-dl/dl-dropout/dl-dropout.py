import numpy as np

def dropout(X: list, mask: list, drop_prob: float, mode: str) -> list:
    """
    Returns the train-mode dropout result or unchanged test input.
    """
    X =  np.asarray(X, dtype = np.float64)
    mask = np.asarray(mask, dtype = np.float64)
    
    if mode == "train":
        X =  X * mask / (1.0 - drop_prob)
    return X