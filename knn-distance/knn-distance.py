import numpy as np

def knn_distance(X_train: list, X_test: list, k: int) -> np.ndarray:
    """
    Returns a NumPy array with shape (n_test, k).
    """
    # Write code here
    X_train = np.asarray(X_train, dtype= np.float64)
    X_test = np.asarray(X_test, dtype= np.float64)
    counter_ = min(k, X_train.shape[0])
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1,1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1,1)
        
    distances = np.sqrt(np.sum((X_test[:, None, :] - X_train[None, : , :] )**2, axis = -1 )  )

    neighbors = np.argsort(distances, axis = -1)[:,:counter_]
    if counter_ < k:
        neighbors = np.concatenate((neighbors , np.full((neighbors.shape[0], k-counter_), -1, dtype = int)), axis = -1)
    
    return neighbors