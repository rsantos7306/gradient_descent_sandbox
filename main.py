import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from gradient_descent.utils import *
from gradient_descent.algorithms import batch_gradient_descent, stochastic_gradient_descent, mini_batch_gradient_descent


def update_plot(event) -> None:
    # reset error data
    errors.clear()
    # assign slider values to variables
    data_a0 = data_a0_slider.val
    data_a1 = data_a1_slider.val
    data_a2 = data_a2_slider.val
    data_a3 = data_a3_slider.val
    data_noise = data_noise_slider.val
    learning_rate = 10 ** (-1 * float(learning_rate_slider.val))
    epochs = int(epochs_slider.val)
    num_points = int(num_points_slider.val)
    init_a0 = init_a0_slider.val
    init_a1 = init_a1_slider.val
    init_a2 = init_a2_slider.val
    init_a3 = init_a3_slider.val
    descent_type = descent_type_button.value_selected

    init_params = [init_a0, init_a1, init_a2, init_a3]
    data_params = [data_a0, data_a1, data_a2, data_a3]

    # set the degree of the fitting function to that of the data
    i = len(data_params) - 1
    while data_params[i] == 0:
        data_params.pop()
        init_params.pop()
        i -= 1

    new_x = np.linspace(-10, 10, num_points)
    new_y = get_fitting_func(new_x, data_params) + data_noise * np.random.randn(num_points)

    # Remove old data points
    for scatter in ax.collections:
        scatter.remove()
    # Plot new data points
    sc = ax.scatter(new_x, new_y, color='blue', label='Original Data Points')

    # Use correct model based on user choice
    match descent_type:
        case 'stochastic':
            params = stochastic_gradient_descent(new_x, new_y, errors=errors, params=init_params,
                                                 learning_rate=learning_rate, epochs=epochs)
        case 'mini batch':
            params = mini_batch_gradient_descent(new_x, new_y, errors=errors, params=init_params,
                                                 batch_size=int(batch_size_slider.val),
                                                 learning_rate=learning_rate, epochs=epochs)
        case 'batch':
            params = batch_gradient_descent(new_x, new_y, errors=errors, params=init_params,
                                            learning_rate=learning_rate, epochs=epochs)
        case _:
            raise Exception("descent type invalid")

    # Tell user to lower learning rate if error value overflows
    if np.any(np.isnan(params)) or np.any(np.isinf(params)):
        show_overflow_NaN_warning()

    # Update best fit line points
    best_fit_func = get_fitting_func(new_x, params)
    line.set_xdata(new_x)
    line.set_ydata(best_fit_func)
    # Re-label the best fit equation and error
    line.set_label(f'Function of best fit: y = {get_func_string(params)}'
                   f'\nError: {calculate_error_mean_squared(new_y, best_fit_func):.3f}')
    ax.set_title(f'Polynomial Regression: {descent_type.capitalize()} Gradient Descent')
    ax.legend()

    # Update and rescale the error graph
    line_errors.set_xdata(list(errors.keys()))
    line_errors.set_ydata(list(errors.values()))
    axes_errors.relim()
    axes_errors.autoscale_view()

    # update the visual graphs
    fig.canvas.draw_idle()
    fig_errors.canvas.draw_idle()


def update_batch_size_max(val) -> None:
    new_max = num_points_slider.val
    batch_size_slider.valmax = new_max
    # set to new maximum if current value would exceed the new maximum
    if batch_size_slider.val > new_max:
        batch_size_slider.val = new_max
    # adjust the slider box to account for new range of values
    batch_size_slider.ax.set_xlim(batch_size_slider.valmin, new_max)
    # redraw
    fig_sliders.canvas.draw_idle()


if __name__ == "__main__":
    # Creates a dictionary for error values for each iteration to be stored in
    errors = {}
    num_points = 100
    # generate x coordinates
    x = np.linspace(-10, 10, num_points)
    # adds noise and creates a general function for data points to follow
    y = get_fitting_func(x, [3.0, 2.0, 1.0]) + np.random.randn(num_points)

    params = [0.0, 0.0, 0.0]

    # Calculates the final coefficients through gradient descent
    params = batch_gradient_descent(x, y, errors=errors, params=params, learning_rate=10.0 ** (-3.5))
    if np.any(np.isnan(params)) or np.any(np.isinf(params)):
        show_overflow_NaN_warning()

    # Generates an ndarray of y-values on the line of best fit given the x-values
    best_fit = get_fitting_func(x, params)
    # output final value
    func_string = get_func_string(params)
    print(f'Function: {func_string} , Error: {calculate_error_mean_squared(y, best_fit)}')

    # initializes the figure for the data points and function of best fit
    fig, ax = plt.subplots(figsize=(10, 8))
    # plots the data points
    sc = ax.scatter(x, y, color='blue', label='Original Data Points')
    # plots the line of best fit, assigns to variable line
    [line] = ax.plot(
        x, best_fit, color='orange',
        label=f'Function of best fit: y = {func_string}'
              f'\nError: {calculate_error_mean_squared(y, best_fit):.3f}'
    )
    # labelling and titling of the figure
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Polynomial Regression: Batch Gradient Descent')
    ax.grid(True)
    ax.legend()

    # initializes the sliders window
    fig_sliders, axes = plt.subplots(13, 1, figsize=(8, 12))
    # gives space for the sliders and buttons
    plt.subplots_adjust(left=0.25, hspace=0.5)

    # initializes the errors graph window
    fig_errors, axes_errors = plt.subplots(figsize=(8, 6))
    [line_errors] = axes_errors.plot(list(errors.keys()), list(errors.values()))
    axes_errors.set_title('Error Vales Over Epochs')
    axes_errors.set_xlabel('Epochs')
    axes_errors.set_ylabel('Mean Squared Error')

    # Sliders for second figure
    data_a0_slider = Slider(axes[0], 'Data a0', -10, 10, valinit=2, valstep=0.1)
    data_a1_slider = Slider(axes[1], 'Data a1', -5, 5, valinit=3, valstep=0.1)
    data_a2_slider = Slider(axes[2], 'Data a2', -5, 5, valinit=1, valstep=0.1)
    data_a3_slider = Slider(axes[3], 'Data a3', -5, 5, valinit=0, valstep=0.1)
    data_noise_slider = Slider(axes[4], 'Data Noise', 0, 20, valinit=1, valstep=0.2)
    learning_rate_slider = Slider(axes[5], 'Learning Rate (1e-val)', 2, 12, valinit=3.5, valstep=0.1)
    epochs_slider = Slider(axes[6], 'Epochs', 20, 10000, valinit=1000, valstep=20)
    num_points_slider = Slider(axes[7], 'Num Points', 10, 500, valinit=100, valstep=10)
    batch_size_slider = Slider(axes[8], 'Batch Size (Mini Batch)', 1, num_points_slider.val, valinit=10, valstep=1)
    init_a0_slider = Slider(axes[9], 'initial a0', -10, 10, valinit=0, valstep=0.1)
    init_a1_slider = Slider(axes[10], 'initial a1', -10, 10, valinit=0, valstep=0.1)
    init_a2_slider = Slider(axes[11], 'initial a2', -10, 10, valinit=0, valstep=0.1)
    init_a3_slider = Slider(axes[12], 'initial a3', -10, 10, valinit=0, valstep=0.1)

    # Change batch size slider if the size of the data set changes
    num_points_slider.on_changed(update_batch_size_max)

    # Allow the user to choose the type of gradient descent
    radio_ax = fig_sliders.add_axes((0.35, 0.06, 0.3, 0.05))
    descent_type_button = RadioButtons(radio_ax, ('stochastic', 'mini batch', 'batch'))

    # Create button to update parameters
    button_ax = fig_sliders.add_axes((0.35, 0.01, 0.3, 0.05))
    update_button = Button(button_ax, 'Update', hovercolor='0.9')
    update_button.on_clicked(update_plot)
    plt.show()
