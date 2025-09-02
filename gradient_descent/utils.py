import numpy as np


def calculate_error_mean_squared(ys: np.ndarray, pred_ys: np.ndarray) -> float:

    """
    Calculate the mean squared error (MSE) of the line of best fit with the data set

    :param ys: data set y-values
    :param pred_ys: predicted y-values with linear regression
    :return: the MSE as a float
    """
    error = 0.0
    # Sums squares of the differences of y pairs between actual and
    for actual_y, pred_y in zip(ys, pred_ys):
        error += (actual_y - pred_y) ** 2
    # Divide by number of pairs to get average
    return error / float(len(ys))
