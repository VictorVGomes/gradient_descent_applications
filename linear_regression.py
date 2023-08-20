import numpy as np

def mse_loss(X, y, weights):
    return ((y - X.dot(weights)) ** 2).mean()

def mse_loss_(y, ypred):
    return ((y - ypred) ** 2).mean()


def lm_grad_f(X, y, weights):
    return -2 * X.T.dot(y) + 2 * (X.T.dot(X)).dot(weights)