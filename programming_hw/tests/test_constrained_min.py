import unittest
import numpy as np
from src.constrained_min import InteriorPointMinimization
from src.utils import plot_iterations, plot_feasible_set_2d, plot_feasible_set_3d

from examples import (
    qp,
    lp,
    qp_ineq_constraint_1,
    qp_ineq_constraint_2,
    qp_ineq_constraint_3,
    lp_ineq_constraint_1,
    lp_ineq_constraint_2,
    lp_ineq_constraint_3,
    lp_ineq_constraint_4,
)


class TestInteriorPointMethod(unittest.TestCase):
    START_POINT_qp = np.array([0.1, 0.2, 0.7], dtype=np.float64)
    START_POINT_lp = np.array([0.5, 0.75], dtype=np.float64)
    minimizer = InteriorPointMinimization()


    def _print_convergence_info(self, x_s, problem_type, constraints):
        x_star = x_s[-1]
        print(f"{problem_type.__name__}, Point of convergence: {x_star}")
        print(f"{problem_type.__name__}, Value at point of convergence: {problem_type(x_star, False)[0]}")

        for i, x in enumerate(x_star):
            print(f"x[{i}] at point of convergence: {x}")

        for i, constraint in enumerate(constraints, start=1):
            constraint_value = constraint(x_star, False)[0]
            print(f"{constraint.__name__} Value at point of convergence: {constraint_value}")

        if problem_type.__name__ == 'qp':
            print(f"Sum of variables at point of convergence: {sum(x_star)}")

    def _test_problem(
            self,
            problem_type,
            start_point,
            constraints,
            eq_constraint_mat,
    ):
        minimizer = self.minimizer
        x_s, obj_values, outer_x_s, outer_obj_values = minimizer.minimize(
            problem_type,
            start_point,
            constraints,
            eq_constraint_mat,
            "wolfe",
            10e-12,
            10e-8,
            100,
            20,
            10e-10,
        )

        self._print_convergence_info(x_s, problem_type, constraints)
        plot_iterations(
            f"Convergence of {problem_type.__name__}",
            obj_values_1=obj_values,
            obj_values_2=outer_obj_values,
            label_1="Inner Objective Value",
            label_2="Outer Objective Value",
        )

        if problem_type.__name__ == "qp":
            plot_feasible_set_3d(np.array(x_s))

        if problem_type.__name__ == "lp":
            plot_feasible_set_2d(np.array(x_s))

    def test_qp(self):
        eq_constraint_mat = np.array([1, 1, 1]).reshape(1, -1)
        constraints = [qp_ineq_constraint_1, qp_ineq_constraint_2, qp_ineq_constraint_3]
        self._test_problem(
            qp,
            self.START_POINT_qp,
            constraints,
            eq_constraint_mat,
        )

    def test_lp(self):
        constraints = [
            lp_ineq_constraint_1,
            lp_ineq_constraint_2,
            lp_ineq_constraint_3,
            lp_ineq_constraint_4,
        ]
        self._test_problem(lp, self.START_POINT_lp, constraints, np.array([]))


if __name__ == "__main__":
    unittest.main()
