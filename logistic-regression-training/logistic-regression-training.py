import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    X = np.asarray(X, dtype = np.float64)
    y = np.asarray(y, dtype = np.float64)
    N, d = X.shape
    
    if y.ndim == 1:
        out_d = 1
        y = y.reshape(-1, 1)
        
    else:
        out_d = y.shape[1]
        

        
    X_aug = np.column_stack([X, np.ones(N)])
    W_aug = np.zeros((d + 1, out_d))
    
    for _ in range(steps):
        logits = X_aug @ W_aug
        preds = _sigmoid(logits)
        grad_W = X_aug.T @ (preds - y)/N
        W_aug = W_aug - lr * grad_W
    
    w = W_aug[:-1, :]
    b = W_aug[-1, :]

    if w.shape[1] == 1:
        w = w[:, 0]       
        b = float(b[0])    
        
    return w,b
        