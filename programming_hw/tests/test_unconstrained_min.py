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

METHOD = "wolfe"
TOLERANCE = 10e-12
STEP_SIZE = 10e-8
MAX_ITER = 100
INITIAL_POINT = np.array([1, 1])
ROSENBROCK_INITIAL_POINT = np.array([-1, 2], dtype=float)


class TestLineSearchMethods(unittest.TestCase):
    gd_minimizer = LineSearchMinimization("Gradient descent")
    newton_minimizer = LineSearchMinimization("Newton")
    line_search_method = METHOD
    tolerance = TOLERANCE
    step_size = STEP_SIZE
    max_iterations = MAX_ITER
    initial_point = INITIAL_POINT
    rosenbrock_initial_point = ROSENBROCK_INITIAL_POINT

    def test_circles(self):
        newton_result = self.newton_minimizer.minimize(
            circles,
            self.initial_point,
            self.line_search_method,
            self.tolerance,
            self.step_size,
            self.max_iterations,
        )
        x_newton, f_x_newton, x_s_newton, obj_values_newton, success_newton = newton_result
        print(
            f"Newton's method - Point of convergence: {x_newton}, "
            f"Objective value: {f_x_newton}, Success: {success_newton}"
        )
        gd_result = self.gd_minimizer.minimize(
            circles,
            self.initial_point,
            self.line_search_method,
            self.tolerance,
            self.step_size,
            self.max_iterations,
        )
        x_gd, f_x_gd, x_s_gd, obj_values_gd, success_gd = gd_result
        print(
            f"Gradient Descent - Point of convergence: {x_gd}, "
            f"Objective value: {f_x_gd}, Success: {success_gd}"
        )
        plot_contours(
            circles,
            "Convergence over Circular Contour Lines",
            x_s_gd,
            x_s_newton,
        )
        plot_iterations(
            "Objective Function Values of Quadratic Function 1 - Circular Contour Lines",
            obj_values_gd,
            obj_values_newton,
            "Gradient Descent",
            "Newton's Method",
        )

    def test_ellipses(self):
        newton_result = self.newton_minimizer.minimize(
            ellipses,
            self.initial_point,
            self.line_search_method,
            self.tolerance,
            self.step_size,
            self.max_iterations,
        )
        x_newton, f_x_newton, x_s_newton, obj_values_newton, success_newton = newton_result
        print(
            f"Newton's Method - Convergence Point: {x_newton}, "
            f"Objective Value: {f_x_newton}, Success: {success_newton}"
        )

        gd_result = self.gd_minimizer.minimize(
            ellipses,
            self.initial_point,
            self.line_search_method,
            self.tolerance,
            self.step_size,
            self.max_iterations,
        )
        x_gd, f_x_gd, x_s_gd, obj_values_gd, success_gd = gd_result
        print(
            f"Gradient Descent - Convergence Point: {x_gd}, "
            f"Objective Value: {f_x_gd}, Success: {success_gd}"
        )

        plot_contours(
            ellipses,
            "Convergence over Elliptical Contour Lines",
            x_s_gd,
            x_s_newton,
        )

        plot_iterations(
            "Objective Function Values of Quadratic Function 2 - Elliptical Contour Lines",
            obj_values_gd,
            obj_values_newton,
            "Gradient Descent",
            "Newton's Method",
        )

    def test_rotated_ellipses(self):
        newton_result = self.newton_minimizer.minimize(
            rotated_ellipses,
            self.initial_point,
            self.line_search_method,
            self.tolerance,
            self.step_size,
            self.max_iterations,
        )
        x_newton, f_x_newton, x_s_newton, obj_values_newton, success_newton = newton_result
        print(
            f"Newton's Method - Convergence Point: {x_newton}, "
            f"Objective Value: {f_x_newton}, Success: {success_newton}"
        )
        gd_result = self.gd_minimizer.minimize(
            rotated_ellipses,
            self.initial_point,
            self.line_search_method,
            self.tolerance,
            self.step_size,
            self.max_iterations,
        )
        x_gd, f_x_gd, x_s_gd, obj_values_gd, success_gd = gd_result
        print(
            f"Gradient Descent - Convergence Point: {x_gd}, "
            f"Objective Value: {f_x_gd}, Success: {success_gd}"
        )
        plot_contours(
            rotated_ellipses,
            "Convergence over Rotated Elliptical Contour Lines",
            x_s_gd,
            x_s_newton,
        )
        plot_iterations(
            "Objective Function Values of Quadratic Function 3 - Rotated Elliptical Contour Lines",
            obj_values_gd,
            obj_values_newton,
            "Gradient Descent",
            "Newton's Method",
        )

    def test_rosenbrock(self):
        max_iterations_gd = 10000
        newton_result = self.newton_minimizer.minimize(
            rosenbrock,
            self.rosenbrock_initial_point,
            self.line_search_method,
            self.tolerance,
            self.step_size,
            self.max_iterations,
        )
        x_newton, f_x_newton, x_s_newton, obj_values_newton, success_newton = newton_result
        print(
            f"Newton's Method - Convergence Point: {x_newton}, "
            f"Objective Value: {f_x_newton}, Success: {success_newton}"
        )

        gd_result = self.gd_minimizer.minimize(
            rosenbrock,
            self.rosenbrock_initial_point,
            self.line_search_method,
            self.tolerance,
            self.step_size,
            max_iterations_gd
        )
        x_gd, f_x_gd, x_s_gd, obj_values_gd, success_gd = gd_result
        print(
            f"Gradient Descent - Convergence Point: {x_gd}, "
            f"Objective Value: {f_x_gd}, Success: {success_gd}"
        )

        plot_contours(
            rosenbrock,
            "Convergence over Rosenbrock Function Contour Lines",
            x_s_gd,
            x_s_newton,
        )

        plot_iterations(
            "Objective Function Values of the Rosenbrock Function",
            obj_values_gd,
            obj_values_newton,
            "Gradient Descent",
            "Newton's Method",
        )

    def test_linear(self):
        gd_result = self.gd_minimizer.minimize(
            linear,
            self.initial_point,
            self.line_search_method,
            self.tolerance,
            self.step_size,
            self.max_iterations,
        )
        x_gd, f_x_gd, x_s_gd, obj_values_gd, success_gd = gd_result
        print(
            f"Gradient Descent - Convergence Point: {x_gd}, "
            f"Objective Value: {f_x_gd}, Success: {success_gd}"
        )

        plot_contours(
            linear,
            "Gradient Descent Convergence Over Linear Function",
            x_s_gd,
        )

        plot_iterations(
            "Objective Function Values During Gradient Descent (Linear Function)",
            obj_values_1=obj_values_gd,
            label_1="Gradient Descent",
        )

    def test_triangles(self):
        newton_result = self.newton_minimizer.minimize(
            triangles,
            self.initial_point,
            self.line_search_method,
            self.tolerance,
            self.step_size,
            self.max_iterations,
        )
        x_newton, f_x_newton, x_s_newton, obj_values_newton, success_newton = newton_result
        print(
            f"Newton's Method - Convergence Point: {x_newton}, "
            f"Objective Value: {f_x_newton}, Success: {success_newton}"
        )

        gd_result = self.gd_minimizer.minimize(
            triangles,
            self.initial_point,
            self.line_search_method,
            self.tolerance,
            self.step_size,
            self.max_iterations,
        )
        x_gd, f_x_gd, x_s_gd, obj_values_gd, success_gd = gd_result
        print(
            f"Gradient Descent - Convergence Point: {x_gd}, "
            f"Objective Value: {f_x_gd}, Success: {success_gd}"
        )

        plot_contours(
            triangles,
            "Convergence Over Smoothed Corner Triangles Contour Lines",
            x_s_gd,
            x_s_newton,
        )

        plot_iterations(
            "Objective Function Values for Smoothed Corner Triangles",
            obj_values_gd,
            obj_values_newton,
            "Gradient Descent",
            "Newton's Method",
        )


if __name__ == "__main__":
    unittest.main()
