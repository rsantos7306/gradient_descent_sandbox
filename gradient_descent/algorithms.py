import numpy as np
from .utils import calculate_error_mean_squared


def update_params(slope: float, intercept: float, x: np.ndarray, y: np.ndarray, learning_rate: float) -> (float, float):

    """
    Perform one gradient descent update step.
    """

    slope_grad = 0.0
    intercept_grad = 0.0
    # handle stochastic case with single value x and y
    if np.isscalar(x):
        n = 1.0
        x = np.array([x])
        y = np.array([y])
    else:
        n = float(len(x))

    for x_point, y_point in zip(x, y):
        # Partial derivative of cost function with respect to slope
        slope_grad += -2 * x_point * (y_point - (x_point * slope + intercept))
        # Partial with respect to intercept
        intercept_grad += -2 * (y_point - (x_point * slope + intercept))

    slope_grad = slope_grad / n
    intercept_grad = intercept_grad / n

    # Make the step to find new parameters, with hyperparameter learning rate controlling step size
    # We want to move towards a local minimum, hence moving against the derivative
    slope = slope - slope_grad * learning_rate
    intercept = intercept - intercept_grad * learning_rate

    return slope, intercept


def batch_gradient_descent(x: np.ndarray, y: np.ndarray, errors: dict, slope=0.0, intercept=0.0, learning_rate=0.001,
                           epochs=1000) -> (
        float, float):
    # hyperparameter epochs for how many times gradient descent step is performed
    for i in range(epochs):
        # add error at this epoch to errors to plot later
        errors[i] = calculate_error_mean_squared(y, (x * slope + intercept))
        slope, intercept = update_params(slope, intercept, x, y, learning_rate)
    return slope, intercept


# Lower learning default learning rate in both mini batch and stochastic to
# prevent outliers from drastically changing parameters
def mini_batch_gradient_descent(x: np.ndarray, y: np.ndarray, errors: dict, slope=0.0, intercept=0.0, batch_size=10,
                                learning_rate=0.001, epochs=1000) -> (
        float, float):
    # hyperparameter iterations for how many times gradient descent step is performed
    for j in range(epochs):
        errors[j] = calculate_error_mean_squared(y, (x * slope + intercept))
        # the addition of batch_size - 1 means that if batch_size doesn't divide evenly, another iteration is performed
        # for the remaining data
        for i in range((len(x) + batch_size - 1) // batch_size):
            # as python slicing is safe, no need to worry about index out of bounds errors for the final batch
            slope, intercept = update_params(slope, intercept, x[i * batch_size:(i + 1) * batch_size],
                                             y[i * batch_size:(i + 1) * batch_size], learning_rate)
    return slope, intercept


def stochastic_gradient_descent(x: np.ndarray, y: np.ndarray, errors: dict, slope=0.0, intercept=0.0,
                                learning_rate=0.001,
                                epochs=1000) -> (float, float):
    for j in range(epochs):
        errors[j] = calculate_error_mean_squared(y, (x * slope + intercept))
        for point_x, point_y in zip(x, y):
            slope, intercept = update_params(slope, intercept, point_x, point_y, learning_rate)

    return slope, intercept
