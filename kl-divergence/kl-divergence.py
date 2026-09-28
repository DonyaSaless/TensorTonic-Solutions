import numpy as np

def kl_divergence(p: list, q: list, eps: float = 1e-12) -> float:
    """
    Returns the divergence as a float.
    """
    # Write code here
    p = np.asarray(p , dtype = np.float64)
    q = np.asarray(q , dtype = np.float64)
    positive = p>0
    q[q == 0] = eps
    elements = np.where(positive, p * np.log(p / q), 0.0)
    return float(np.sum(elements))