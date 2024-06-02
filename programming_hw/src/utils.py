import numpy as np
import matplotlib.pyplot as plt
import warnings

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
    CS = ax.contour(X, Y, Z, levels=contour_levels, cmap="viridis")
    ax.clabel(CS, inline=True, fontsize=8)

    if xy_gd is not None:
        xy_gd = np.array(xy_gd)
        ax.plot(xy_gd[:, 0], xy_gd[:, 1], "o-", label="Gradient Descent", color="red")

    if xy_newton is not None:
        xy_newton = np.array(xy_newton)
        ax.plot(xy_newton[:, 0], xy_newton[:, 1], "x-", label="Newton's Method", color="blue")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")
    plt.show()


def plot_iterations(
        title,
        obj_values_1=None,
        obj_values_2=None,
        label_1=None,
        label_2=None,
):
    """
    Plots the objective function values over iterations for two methods.
    :param title:
    :param obj_values_1:
    :param obj_values_2:
    :param label_1:
    :param label_2:
    :return:
    """
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


import numpy as np
import matplotlib.pyplot as plt


def plot_feasible_set_2d(path_points=None):
    """
    Plots the feasible region for a given 2D problem and the path points
    of an algorithm if provided.

    Parameters:
    - path_points: List of tuples [(x1, y1), (x2, y2), ...] representing
                   the path points of an algorithm. Default is None.
    """
    d = np.linspace(-2, 4, 300)
    x, y = np.meshgrid(d, d)

    plt.imshow(
        ((y >= -x + 1) & (y <= 1) & (x <= 2) & (y >= 0)).astype(int),
        extent=(x.min(), x.max(), y.min(), y.max()),
        origin="lower",
        cmap="Greys",
        alpha=0.3,
    )

    x_line = np.linspace(0, 4, 2000)
    y1 = -x_line + 1
    y2 = np.ones(x_line.size)
    y3 = np.zeros(x_line.size)
    x_boundary = np.ones(x_line.size) * 2

    plt.plot(x_line, y1, 'b-', label=r"$y = -x + 1$")
    plt.plot(x_line, y2, 'g-', label=r"$y = 1$")
    plt.plot(x_line, y3, 'r-', label=r"$y = 0$")
    plt.plot(x_boundary, x_line, 'm-', label=r"$x = 2$")

    x_path, y_path = zip(*path_points)
    plt.plot(x_path, y_path, label="Algorithm's Path", color="k", marker=".", linestyle="--")

    plt.xlim(0, 3)
    plt.ylim(0, 2)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title("Feasible Region")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.0)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_feasible_set_3d(path_points=None):
    """
    Plots the feasible region for a given 3D problem and the path points
    of an algorithm if provided.

    Parameters:
    - path_points: numpy array of shape (n, 3) representing the path points
                   of an algorithm. Default is None.
    """
    d = np.linspace(-2, 4, 100)
    x, y = np.meshgrid(d, d)
    z = x + y

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, alpha=0.5, cmap='viridis')

    if path_points is not None and path_points.shape[1] == 3:
        ax.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], 'o-', color='red',
                label="Algorithm's Path")

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Feasible Region in 3D')
    ax.legend()
    plt.show()