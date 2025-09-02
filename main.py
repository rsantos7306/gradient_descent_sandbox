import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from gradient_descent.utils import calculate_error_mean_squared
from gradient_descent.algorithms import batch_gradient_descent, stochastic_gradient_descent, mini_batch_gradient_descent


def update_plot(event) -> None:
    # reset error data
    errors.clear()
    data_slope = data_slope_slider.val
    data_intercept = data_intercept_slider.val
    data_noise = data_noise_slider.val
    learning_rate = learning_rate_slider.val
    epochs = int(epochs_slider.val)
    num_points = int(num_points_slider.val)
    init_slope = init_slope_slider.val
    init_int = init_int_slider.val
    descent_type = descent_type_button.value_selected

    new_x = np.linspace(0, 10, num_points)
    new_y = data_slope * new_x + data_intercept + data_noise * np.random.randn(num_points)

    # Remove old data points
    for scatter in ax.collections:
        scatter.remove()
    # Plot new data points
    sc = ax.scatter(new_x, new_y, color='blue', label='Original Data Points')

    # Use correct model based on user choice
    match descent_type:
        case 'stochastic':
            slope, intercept = stochastic_gradient_descent(new_x, new_y, errors=errors, slope=init_slope,
                                                           intercept=init_int,
                                                           learning_rate=learning_rate, epochs=epochs)
        case 'mini batch':
            slope, intercept = mini_batch_gradient_descent(new_x, new_y, errors=errors, slope=init_slope,
                                                           intercept=init_int,
                                                           batch_size=int(batch_size_slider.val),
                                                           learning_rate=learning_rate, epochs=epochs)
        case 'batch':
            slope, intercept = batch_gradient_descent(new_x, new_y, errors=errors, slope=init_slope, intercept=init_int,
                                                      learning_rate=learning_rate, epochs=epochs)
        case _:
            raise Exception("descent type invalid")

    # Update best fit line points
    line.set_xdata(new_x)
    line.set_ydata(new_x * slope + intercept)
    # Re-label the slope, intercept and error
    line.set_label(f'Line of best fit: y = {slope:.2f}x + {intercept:.2f}'
                   f'\nError: {calculate_error_mean_squared(new_y, new_x * slope + intercept):.3f}')
    ax.set_title(f'Linear Regression: {descent_type.capitalize()} Gradient Descent')
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
    x = np.linspace(0, 10, num_points)
    # adds noise and creates linear relationship
    y = 2 * x + 3 + np.random.randn(num_points)

    # Calculates the final slope and intercept given the initial parameters
    slope, intercept = batch_gradient_descent(x, y, errors=errors, )

    # Generates an ndarray of y-values on the line of best fit given the x-values
    best_fit = x * slope + intercept
    # output final value
    print(f'Slope: {slope}, Intercept: {intercept}, Error: {calculate_error_mean_squared(y, best_fit)}')

    # initializes the figure for the data points and line of best fit
    fig, ax = plt.subplots(figsize=(10, 8))
    # plots the data points
    sc = ax.scatter(x, y, color='blue', label='Original Data Points')
    # plots the line of best fit, assigns to variable line
    [line] = ax.plot(
        x, best_fit, color='orange',
        label=f'Line of best fit: y = {slope:.2f}x + {intercept:.2f}'
              f'\nError: {calculate_error_mean_squared(y, best_fit):.3f}'
    )
    # labelling and titling of the figure
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Linear Regression: Batch Gradient Descent')
    ax.grid(True)
    ax.legend()

    # initializes the sliders window
    fig_sliders, axes = plt.subplots(9, 1, figsize=(6, 12))
    # gives space for the sliders and buttons
    plt.subplots_adjust(hspace=0.5)

    # initializes the errors graph window
    fig_errors, axes_errors = plt.subplots(figsize=(8, 6))
    [line_errors] = axes_errors.plot(list(errors.keys()), list(errors.values()))
    axes_errors.set_title('Error Vales Over Epochs')
    axes_errors.set_xlabel('Epochs')
    axes_errors.set_ylabel('Mean Squared Error')

    # Sliders for second figure
    data_slope_slider = Slider(axes[0], 'Data Slope', -10, 10, valinit=2, valstep=0.1)
    data_intercept_slider = Slider(axes[1], 'Data Intercept', -5, 5, valinit=3, valstep=0.1)
    data_noise_slider = Slider(axes[2], 'Data Noise', 0, 20, valinit=1, valstep=0.2)
    learning_rate_slider = Slider(axes[3], 'Learning Rate', 0, 0.03, valinit=0.001, valstep=0.001)
    epochs_slider = Slider(axes[4], 'Epochs', 20, 10000, valinit=1000, valstep=20)
    num_points_slider = Slider(axes[5], 'Num Points', 10, 500, valinit=100, valstep=10)
    init_slope_slider = Slider(axes[6], 'Init Slope', -10, 10, valinit=0, valstep=0.1)
    init_int_slider = Slider(axes[7], 'Init Intercept', -5, 5, valinit=0, valstep=0.1)
    batch_size_slider = Slider(axes[8], 'Batch Size (Mini Batch)', 1, num_points_slider.val, valinit=10, valstep=1)

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
