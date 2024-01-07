# gradient_descent_applications
Defines a class in python with a (not optimized) version of gradient descent, in addition to line search which improves the method's convergence capabilities.

Using the class Gradient_Descent available in the gradient_descent.py file, one can generate the following gifs, which demonstrates the power of the method. In addition, line search is used to improve the methods convergence speed, as well as its convergence rate.

# Linear Regression via Gradient Descent

Gif #1 shows the gradient descent method without line search:

![](https://github.com/VictorVGomes/gradient_descent_applications/blob/main/gifs/linear_regression_gradient_descent.gif?raw=true)


Gif #2 shows the gradient descent method with line search:

![](https://github.com/VictorVGomes/gradient_descent_applications/blob/main/gifs/linear_regression_gradient_descent_with_line_search.gif?raw=true)


Both gifs show the methods applied to a simple linear regression problem, but it can be generalized to any problem with a weight/vector matrix and a weight gradient function. 


# Logistic Regression via Maximum Likelihood Estimation using Gradient Descent

The next gif shows gradient descent (without line search) for the purpose of finding the MLE for a simple logistic regression.

Derivation is made using the Bernoulli distribution and maximizing the likelihood of the distribution with respect to some set of parameters $\mathbf{\theta}$.

![](https://github.com/VictorVGomes/gradient_descent_applications/blob/main/gifs/logistic_regression_gradient_descent.gif?raw=true)

