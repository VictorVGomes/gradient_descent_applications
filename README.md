# gradient_descent_applications
Defines a class in python with a (not optimized) version of gradient descent, in addition to line search which improves the method's convergence capabilities.

Using the class Gradient_Descent available in the gradient_descent.py file, one can generate the following gifs, which demonstrates the power of the method. In addition, line search is used to improve the methods convergence speed, as well as its convergence rate.

Gif #1 shows the gradient descent method without line search:

![](https://github.com/gradient_descent_applications/gifs/gradient_descent.gif)


Gif #2 shows the gradient descent method with line search:

![](https://github.com/gradient_descent_applications/gifs/gradient_descent_with_line_search.gif)


Both gifs show the methods applied to a simple linear regression problem, but it can be generalized to any problem with a weight vector/matrix and a weight gradient function. 

As next steps, the method will be applied to the Logistic Regression problem, in the binary response case.