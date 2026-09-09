import numpy as np
import torch
def apply_causal_mask(scores: list, mask_value: float = -1e9) -> np.ndarray:
    """
    Returns a causally masked NumPy array matching the shape of scores.
    """
    # Write code here
    scores_array = np.asarray(scores, dtype = float)
    
    if scores_array.ndim < 2:
        raise ValueError("scores must have at least two dimensions")

    if scores_array.shape[-2] != scores_array.shape[-1]:
        raise ValueError("the final two dimensions must be [T, T]")

    allowed = np.tril(np.ones ((scores_array.shape[-1], scores_array.shape[-1]) , dtype = bool))

    return np.where(allowed, scores_array, mask_value)
    
    