import warnings
import numpy as np
import matplotlib.pyplot as plt


def plot_contours(f, title, xy_gd=None, xy_newton=None, contour_levels=40, grid_size=300):
    """
    Plots contour lines for a given function `f` and overlays optional points
    for gradient descent and Newton's method.
    :param f: A callable function that takes a numpy array [x, y] and returns a scalar.
    :param title: The title for the plot.
    :param xy_gd: (Optional) List of gradient descent points [x, y].
    :param xy_newton: (Optional) List of Newton's method points [x, y].
    :param contour_levels: (Optional) Number of contour levels.
    :param grid_size: (Optional) Number of points for the x and y axis.
    """
    if xy_gd is not None or xy_newton is not None:
        all_points = []
        if xy_gd is not None:
            all_points.append(np.array(xy_gd))
        if xy_newton is not None:
            all_points.append(np.array(xy_newton))

        combined_points = np.concatenate(all_points)
        x_min, x_max = np.min(combined_points[:, 0]), np.max(combined_points[:, 0])
        y_min, y_max = np.min(combined_points[:, 1]), np.max(combined_points[:, 1])

        padding = 1
        x_range = (x_min - padding, x_max + padding)
        y_range = (y_min - padding, y_max + padding)
    else:
        if "linear" in title.lower():
            x_range = (-100, 10)
            y_range = (-200, 10)
        else:
            x_range = (-5, 5)
            y_range = (-5, 5)

    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                function_value = f(np.array([X[i, j], Y[i, j]]))[0]
                Z[i, j] = float(function_value)
            except Exception as e:
                raise ValueError(
                    "Error in `plot_contours`: Ensure `f` returns a scalar value.") from e

    fig, ax = plt.subplots(figsize=(8, 6))
    CS = ax.contour(X, Y, Z, levels=contour_levels, cmap='viridis')
    ax.clabel(CS, inline=True, fontsize=8)

    if xy_gd is not None:
        xy_gd = np.array(xy_gd)
        ax.plot(xy_gd[:, 0], xy_gd[:, 1], 'o-', label='Gradient Descent', color='red')

    if xy_newton is not None:
        xy_newton = np.array(xy_newton)
        ax.plot(xy_newton[:, 0], xy_newton[:, 1], 'x-', label="Newton's Method", color='blue')

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc='upper right')

    # Display the plot
    plt.show()


def plot_iterations(
        title,
        obj_values_1=None,
        obj_values_2=None,
        label_1=None,
        label_2=None,
):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective Function Value")

    if obj_values_1 is not None:
        valid_values_1 = [v for v in obj_values_1 if v is not None]
        ax.plot(range(len(valid_values_1)), valid_values_1, label=label_1)

    if obj_values_2 is not None:
        valid_values_2 = [v for v in obj_values_2 if v is not None]
        ax.plot(range(len(valid_values_2)), valid_values_2, label=label_2)

    ax.legend()
    plt.show()
