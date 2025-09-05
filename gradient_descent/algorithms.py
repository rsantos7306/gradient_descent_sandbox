import numpy as np
from .utils import calculate_error_mean_squared, calculate_partial_deriv, get_fitting_func


def update_params(params: [float], x: np.ndarray, y: np.ndarray, learning_rate: float) -> None:
    """
    Perform one gradient descent update step.
    """

    # Initialize a gradient for each coefficient
    grads = [0.0] * len(params)
    # handle stochastic case with single value x and y
    if np.isscalar(x):
        n = 1.0
        x = np.array([x])
        y = np.array([y])
    else:
        n = float(len(x))

    for x_point, y_point in zip(x, y):
        for i in range(len(grads)):
            grads[i] += calculate_partial_deriv(x_point, y_point, params, i, n)

    # Make the step to find new parameters, with hyperparameter learning rate controlling step size
    # We want to move towards a local minimum, hence moving against the derivative

    for i in range(len(params)):
        params[i] -= grads[i] * learning_rate


def batch_gradient_descent(x: np.ndarray, y: np.ndarray, errors: dict, params: [float], learning_rate=0.004,
                           epochs=1000) -> [float]:
    # hyperparameter epochs for how many times gradient descent step is performed on the whole data set

    for i in range(epochs):
        fitting_function = get_fitting_func(x, params)
        # add error at this epoch to errors to plot later
        errors[i] = calculate_error_mean_squared(y, fitting_function)
        update_params(params, x, y, learning_rate)
    return params


# Lower learning default learning rate in both mini batch and stochastic to
# prevent outliers from drastically changing parameters
def mini_batch_gradient_descent(x: np.ndarray, y: np.ndarray, errors: dict, params: [float], batch_size=10,
                                learning_rate=0.001, epochs=1000) -> [float]:
    for j in range(epochs):
        fitting_function = get_fitting_func(x, params)
        errors[j] = calculate_error_mean_squared(y, fitting_function)
        # the addition of batch_size - 1 means that if batch_size doesn't divide evenly, another iteration is performed
        # for the remaining data
        for i in range((len(x) + batch_size - 1) // batch_size):
            # as python slicing is safe, no need to worry about index out of bounds errors for the final batch
            update_params(params, x[i * batch_size:(i + 1) * batch_size],
                          y[i * batch_size:(i + 1) * batch_size], learning_rate)
    return params


def stochastic_gradient_descent(x: np.ndarray, y: np.ndarray, errors: dict, params: [float],
                                learning_rate=0.001, epochs=1000) -> [float]:
    for j in range(epochs):
        fitting_function = get_fitting_func(x, params)
        errors[j] = calculate_error_mean_squared(y, fitting_function)
        for point_x, point_y in zip(x, y):
            update_params(params, point_x, point_y, learning_rate)

    return params
