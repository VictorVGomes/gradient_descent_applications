import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib, matplotlib.pyplot as plt, os
import imageio
matplotlib.use("Agg")

class Gradient_Descent:
    def __init__(
        self,
        weights,
        gradient_function,
        loss_function=None,
        epsilon=1e-6,
    ):
        """-- Constructor"""
        self.weights = weights
        self.gradient_function = gradient_function
        self.loss_function = loss_function
        self.epsilon = epsilon
        self.reached_convergence = False
        self.steps_until_convergence = 0

    def line_search(
        self,
        current_gradients,
    ):
        """Defines the line search algorithm
        used to estimate the step size for
        the current gradient step.
        """
        alpha = 1.0  ### initialized as 1, but can be lower

        loss_f = lambda w: self.loss_function(
            X=self.X,
            y=self.y,
            weights=w,
        )
        cgrad_dot = current_gradients.T.dot(current_gradients)
        lw1 = loss_f(self.weights - alpha * current_gradients)
        lw2 = loss_f(w=self.weights) - alpha**2 * current_gradients.T.dot(
            current_gradients
        )

        while not (lw1 <= lw2):
            alpha *= self.tau  ### Update the current alpha, as stated above

            lw1 = loss_f(self.weights - alpha * current_gradients)
            lw2 = loss_f(w=self.weights) - alpha**2 * cgrad_dot

        return alpha

    def update_weights(
        self,
    ):
        """Updates the current model weights
        through gradient descent + line search methods
        """
        self.grad_weights = self.gradient_function(
            X=self.X,
            y=self.y,
            weights=self.weights,
        )

        if self.loss_function:
            alpha = self.line_search(
                current_gradients=self.grad_weights,
            )
        else:
            alpha = 1e-5

        self.weights -= alpha * self.grad_weights

    def gradient_norm(
        self,
    ):
        current_gradient_norm = np.linalg.norm(self.grad_weights)
        return current_gradient_norm

    def step(
        self,
    ):
        ### the step consists of updating the weights through gradient descent
        ### and then checking if the norm is close enough to zero
        self.update_weights()
        self.steps_until_convergence += 1

        if self.gradient_norm() <= self.epsilon:
            self.reached_convergence = True

    def fit(self, X, y, tau=95 / 100):
        ###
        self.X = X
        self.y = y
        self.tau = tau
        self.weights_by_step = [self.weights.copy()]
        while not self.reached_convergence:
            self.step()
            self.weights_by_step.append(self.weights.copy())

        print(f"Reached convergence after {self.steps_until_convergence} steps.")
        return self.weights

