import numpy as np, os, shutil, imageio, matplotlib.pyplot as plt
from time import sleep

class Gradient_Descent:
    def __init__(
        self,
        weights,
        gradient_function,
        loss_function=None,
        epsilon=1e-6,
        link=None,
    ):
        """-- Constructor --"""
        self.weights = weights
        self.gradient_function = gradient_function
        self.loss_function = loss_function
        self.epsilon = epsilon
        self.reached_convergence = False
        self.steps_until_convergence = 0
        self.weights_by_step = [weights.copy()]
        self.break_ = False
        self.link = link

    def line_search(
        self,
        current_gradients,
    ):
        """ Defines the line search algorithm
            used to estimate the step size for
            the current gradient step.
        """
        alpha = 1.0  ### initialized as 1, but can be lower

        loss_f = lambda w: self.loss_function(
            X=self.X,
            y=self.y,
            weights=w,
        )
        cgrad_dot = -current_gradients.T.dot(current_gradients)
        lw1 = loss_f(self.weights - alpha * current_gradients)
        lw2 = loss_f(w=self.weights) + alpha ** 2 * cgrad_dot

        while not (lw1 <= lw2):
            alpha *= self.tau  ### Update the current alpha, as stated above

            lw1 = loss_f(self.weights - alpha * current_gradients)
            lw2 = loss_f(w=self.weights) + alpha ** 2 * cgrad_dot

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
            self.weights -= alpha * self.grad_weights
        else:
            self.weights -= self.alpha * self.grad_weights

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
        
        if self.iterations:
            if self.steps_until_convergence >= self.iterations:
                self.break_ = True

        if self.gradient_norm() <= self.epsilon:
            self.reached_convergence = True

    def fit(self, X, y, tau=95 / 100, alpha=1e-5, iterations=None):
        ###
        self.X = X
        self.y = y
        self.tau = tau
        self.iterations = iterations
        self.alpha = alpha

        while not (self.reached_convergence or self.break_):
            self.step()
            self.weights_by_step.append(self.weights.copy())

        print(f"Reached convergence after {self.steps_until_convergence} steps.")
        return self.weights

    def predict(self, X, weights=None):
        ###
        link = self.link        
        if weights is not None:
            ypred = X.dot(weights)
        else:
            ypred = X.dot(self.weights)

        if link is not None:
            ypred = link(ypred)

        return ypred
        


def generate_gif(
    X:np.ndarray,
    y:np.ndarray,
    lossf,
    w:'fitted gradient_descent obj.',
    gif_name:str, n:int=5_000,
    duration=0.1, ):

    # make temp dir
    if not os.path.isdir('temp_dir'):
        os.makedirs('temp_dir')    

    params = np.hstack(w.weights_by_step).T

    X_1 = np.concatenate(
        (
        np.ones((n, 1)),
        np.linspace(X[:, 1].min(), X[:, 1].max(), num=n).reshape(-1, 1),
        ),
        axis=1,
    )


    for i, (beta0, beta1) in enumerate(params):

        weights = np.concatenate((np.array([beta0]), np.array([beta1]))).reshape(2, 1)
        
        ypred_t = w.predict(X, weights)
        loss_i = np.round(lossf(y, ypred_t), 2)  
        ypred_s = w.predict(X_1, weights)

        xx, ypr = zip(*sorted(zip(X_1[:, 1], ypred_s.reshape(-1,))))
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        plt.title(f'X1 Vs y, iteration {i}, loss={loss_i}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(X_1[:, 1].min()-1, X_1[:, 1].max()+1)
        plt.ylim(y.min()-0.1, y.max()+0.1)

        plt.scatter(X[:, 1], y, s=7, label='True y')
        plt.plot(xx, ypr, alpha=0.8, linewidth=4, c='red', label='estimated f(x)=y')
        
        plt.legend()
        fig.savefig(f'temp_dir/temp_fig_{i}.png')
        plt.close()

    figs = os.listdir('temp_dir')
    figs.sort(key=lambda s: int(s.split('.')[0].split('_')[-1]))
    figs = ['temp_dir/'+f for f in figs]
    frames = [imageio.imread(file_) for file_ in figs]

    imageio.mimsave(
        f"gifs/{gif_name}.gif",
        frames,
        "GIF",
        duration=duration,
        loop=1_000,
    )

    shutil.rmtree('temp_dir')
    


    

