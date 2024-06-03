import numpy as np
import math


class InteriorPointMinimization:
    WOLFE_COND_CONST = 0.01
    BACKTRACKING_CONST = 0.5
    T = 1
    MU = 10

    def __wolfe(self, f, p, x):
        alpha = 1.0
        f_x_0 = f(x, False)[0]
        dot_grad = np.dot(f(x, False)[1], p)
        while (f(x + alpha * p, False)[0] >
               f_x_0 + self.WOLFE_COND_CONST * alpha * dot_grad):
            alpha *= self.BACKTRACKING_CONST
            if alpha < 1e-6:
                break
        return alpha

    def phi(self, constraints, x):
        f, g, h = 0, 0, 0
        for func in constraints:
            f_x, g_x, h_x = func(x, True)
            f += math.log(-f_x)
            grad = g_x / f_x
            g += grad
            grad_mesh = np.tile(
                grad.reshape(grad.shape[0], -1), (1, grad.shape[0])
            ) * np.tile(grad.reshape(grad.shape[0], -1).T, (grad.shape[0], 1))
            h += (h_x * f_x - grad_mesh) / f_x ** 2

        return -f, -g, -h

    def construct_block_matrix(self,hessian, equality_constraints_matrix):
        if equality_constraints_matrix.size:
            upper_block = np.concatenate([hessian, equality_constraints_matrix.T], axis=1)
            zero_block_shape = (
                equality_constraints_matrix.shape[0],
                equality_constraints_matrix.shape[0],
            )
            lower_block = np.concatenate(
                [equality_constraints_matrix, np.zeros(zero_block_shape)],
                axis=1,
            )
            block_matrix = np.concatenate([upper_block, lower_block], axis=0)
        else:
            block_matrix = hessian
        return block_matrix

    def minimize(
        self,
        objective_func,
        initial_guess,
        inequality_constraints,
        equality_constraints_matrix,
        step_length,
        objective_tolerance,
        parameter_tolerance,
        max_inner_iterations,
        max_outer_iterations,
        tolerance_epsilon,
    ):
        current_x = initial_guess
        objective_value, gradient, hessian = objective_func(current_x, True)
        phi_value, phi_gradient, phi_hessian = self.phi(inequality_constraints, current_x)
        t = self.T

        x_history, outer_x_history = [initial_guess], [initial_guess]
        objective_values, outer_objective_values = [objective_value], [objective_value]

        objective_value = t * objective_value + phi_value
        gradient = t * gradient + phi_gradient
        hessian = t * hessian + phi_hessian

        for outer_iter in range(max_outer_iterations):
            block_matrix = self.construct_block_matrix(hessian, equality_constraints_matrix)
            eq_vector = np.concatenate([-gradient, np.zeros(block_matrix.shape[0] - len(gradient))])

            previous_x = current_x
            previous_objective_value = objective_value

            for inner_iter in range(max_inner_iterations):
                if inner_iter != 0 and np.sum(np.abs(current_x - previous_x)) < parameter_tolerance:
                    break

                search_direction = np.linalg.solve(block_matrix, eq_vector)[: len(current_x)]
                lambda_value = np.sqrt(
                    np.dot(search_direction.T, np.dot(hessian, search_direction))
                )

                if 0.5 * lambda_value ** 2 < objective_tolerance or \
                        (
                                inner_iter != 0 and
                                previous_objective_value - objective_value < objective_tolerance
                        ):
                    break

                alpha = self.__wolfe(
                    objective_func,
                    search_direction,
                    current_x,
                ) if step_length == "wolfe" else step_length

                previous_x = current_x
                previous_objective_value = objective_value

                current_x = current_x + alpha * search_direction
                objective_value, gradient, hessian = objective_func(current_x, True)
                phi_value, phi_gradient, phi_hessian = self.phi(inequality_constraints, current_x)

                x_history.append(current_x)
                objective_values.append(objective_value)

                objective_value = t * objective_value + phi_value
                gradient = t * gradient + phi_gradient
                hessian = t * hessian + phi_hessian

            outer_x_history.append(current_x)
            outer_objective_values.append((objective_value - phi_value) / t)

            if len(inequality_constraints) / t < tolerance_epsilon:
                return x_history, objective_values, outer_x_history, outer_objective_values

            t *= self.MU

        return x_history, objective_values, outer_x_history, outer_objective_values



