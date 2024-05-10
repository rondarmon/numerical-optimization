import unittest
import numpy as np

from src.unconstrained_min import LineSearchMinimization
from src.utils import plot_contours, plot_iterations
from tests.examples import (
    circles,
    ellipses,
    rotated_ellipses,
    rosenbrock,
    linear,
    triangles,
)


INITIAL_POINT = np.array([1, 1]).transpose()
ROSENBROCK_INITIAL_POINT = np.array([-1, 2], dtype=float).transpose()
LINE_SEARCH_METHOD = "wolfe"
TOLERANCE = 10e-12
STEP_SIZE = 10e-8
MAX_ITERATIONS = 100
MAX_ITERATIONS_GD_ROSENBROCK = 10000
MAX_ITERATIONS_NEWTON_ROSENBROCK = 100


class TestLineSearchMethods(unittest.TestCase):
    def setUp(self):
        self.gd_minimizer = LineSearchMinimization("Gradient descent")
        self.newton_minimizer = LineSearchMinimization("Newton")

    def run_test(self, function, title, initial_point, max_iterations=None):
        if max_iterations is None:
            max_iterations = MAX_ITERATIONS

        results = {
            "Gradient Descent": self.gd_minimizer.minimize(
                function,
                initial_point,
                LINE_SEARCH_METHOD,
                TOLERANCE,
                STEP_SIZE,
                max_iterations,
            ),
            "Newton's Method": self.newton_minimizer.minimize(
                function,
                initial_point,
                LINE_SEARCH_METHOD,
                TOLERANCE, STEP_SIZE,
                max_iterations,
            ),
        }

        for method, result in results.items():
            x, f_x, x_s, obj_values, success = result
            print(
                f"{method} - Convergence Point: {x}, "
                f"Objective Value: {f_x}, Success: {success}"
            )

        plot_contours(
            function,
            f"Convergence over {title}",
            results["Gradient Descent"][2],
            results["Newton's Method"][2],
        )

        plot_iterations(
            f"Objective Function Values for {title}",
            results["Gradient Descent"][3],
            results["Newton's Method"][3],
            "Gradient Descent",
            "Newton's Method",
        )

    def test_circles(self):
        self.run_test(circles, "Circular Contour Lines", INITIAL_POINT)

    def test_ellipses(self):
        self.run_test(ellipses, "Elliptical Contour Lines", INITIAL_POINT)

    def test_rotated_ellipses(self):
        self.run_test(rotated_ellipses, "Rotated Elliptical Contour Lines", INITIAL_POINT)

    def test_rosenbrock(self):
        self.run_test(
            rosenbrock,
            "Rosenbrock Function Contour Lines",
            ROSENBROCK_INITIAL_POINT,
            max_iterations=MAX_ITERATIONS_GD_ROSENBROCK,
        )

    def test_linear(self):
        self.run_test(linear, "Linear Function", INITIAL_POINT)

    def test_triangles(self):
        self.run_test(triangles, "Smoothed Corner Triangles", INITIAL_POINT)


if __name__ == "__main__":
    unittest.main()
