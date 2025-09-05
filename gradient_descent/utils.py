import numpy as np
from matplotlib.widgets import Slider
import tkinter as tk
from tkinter import messagebox


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


def calculate_partial_deriv(x_point: float, y_point: float, params: [float], respect_variable: int, n: float) -> float:
    return (-2 * (x_point ** (respect_variable)) * (y_point - float(np.polyval(list(reversed(params)), x_point)))) / n


def get_fitting_func(x: np.ndarray, params: list[float]) -> np.ndarray:
    return np.polyval(list(reversed(params)), x)


def get_func_string(params):
    func_string = ""
    for i in range(len(params) - 1, 0, -1):
        func_string += f"{params[i]:.2f}x^{i} + "
    func_string += f"{params[0]:.2f}"
    return func_string


def create_coef_slider(ax, i, val_init=0.0):
    slider = Slider(ax=ax, label=f'a{i}', valmin=-10, valmax=10, valinit=val_init, valstep=0.1)
    return slider


def show_overflow_NaN_warning():
    # Create hidden root window
    root = tk.Tk()
    root.withdraw()  # Hide the main Tk window

    messagebox.showerror(
        "Overflow Detected",
        "⚠ Gradient descent produced NaN or Inf values!\n"
        "Try decreasing the learning rate by increasing the slider."
    )

    root.destroy()
