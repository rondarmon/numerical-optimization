import numpy as np
import matplotlib.pyplot as plt
import warnings


def plot_contours(f, title, xy_gd=None, xy_newton=None):
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]], dtype=float), False)[0]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    CS = ax.contour(X, Y, Z, 20)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


    if xy_gd is not None:
        ax.plot(
            [x[0] for x in xy_gd], [x[1] for x in xy_gd],
            label="Gradient Descent",
        )

    if xy_newton is not None:
        ax.plot(
            [x[0] for x in xy_newton],
            [x[1] for x in xy_newton],
            label="Newton's Method",
        )

    ax.legend(["Gradient Descent", "Newton's Method"])
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


def plot_feasible_set_2d(path_points):
    d = np.linspace(-2, 4, 300)
    x, y = np.meshgrid(d, d)
    z = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i, j] = 2 * x[i, j] + y[i, j]
    plt.contourf(x, y, z, levels=[-10, 0, 10], colors=["b", "r"], alpha=0.2)

    plt.plot(path_points[:, 0], path_points[:, 1],)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Feasible Set and Path")
    plt.show()


def plot_feasible_set_3d(path_points):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        d = np.linspace(-2, 4, 300)
        x, y = np.meshgrid(d, d)
        z = np.zeros(x.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                z[i, j] = 2 * x[i, j] + y[i, j]
        ax.plot_surface(x, y, z, alpha=0.2)

        # plot the path
        ax.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2])
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")
        ax.set_title("Feasible Set and Path")
        plt.show()
