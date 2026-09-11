import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Returns the position-wise feed-forward output.
    """
    hidden1 = (x @W1) + b1
    ReLU = np.maximum(0,hidden1)
    hidden2 = np.array((ReLU @W2) + b2)
    return hidden2
    
    