import numpy as np

def angle_between_3d(v: list, w: list) -> float:
    """
    Returns the angle as a float.
    """
    # Write code here
    v = np.asarray(v)
    w = np.asarray(w)
    c = np.dot(v,w)/(np.sqrt(v@v) * np.sqrt(w@w))
    return np.arccos(c)