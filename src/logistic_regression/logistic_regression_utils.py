import numpy as np

def logistic_gradient_f(X, y, weights, newtons_m=False):
    # function will include the hessian for convergence purposes
    n = y.shape[0]
    et = np.exp(-X.dot(weights))
    et = 1 / (1 + et)
    grad = X.T.dot((y - et))
    # the minus sign here represents the
    # fact that we are actually trying to
    # find the maximum, since the derivation
    # used for the betas is by maximum likelihood estimation,
    # so we are actually climbing up the ladder.
    # the minus only works in this context because the gradient descent
    # class is for loss function minimization, so it, by default, subtracts
    # the gradient from the current betas.
    
    if newtons_m:
        X_ = X * ( et / (1 + et) )
        hess_1 = np.linalg.inv(X.T.dot(X_))
        return -hess_1.dot(grad)
    return -grad / n

def log_loss(y_true, y_pred):
    eps = 1e-20
    return np.abs((y_true * np.log(y_pred + eps) + (1-y_true) * np.log(1-y_pred + eps)).sum())

def logistic_link(y):
    return 1 / (1 + np.exp(-y))
