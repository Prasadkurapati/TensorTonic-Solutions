import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    N, D = X.shape
    
    w = np.zeros(D)
    b = 0.0
    
    for _ in range(steps):
        z = X @ w + b
        p = _sigmoid(z)
        
        dw = X.T @ (p - y) / N
        db = np.mean(p - y)
        
        w -= lr * dw
        b -= lr * db
    
    return w, b
    pass