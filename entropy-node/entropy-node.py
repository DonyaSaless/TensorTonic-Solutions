import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    y = np.asarray(y, dtype = int)
    _ , counter_ = np.unique(y, return_counts = True)
    counter_ = counter_ / len(y)
    return float(- np.sum(counter_ * np.log2(counter_ )))