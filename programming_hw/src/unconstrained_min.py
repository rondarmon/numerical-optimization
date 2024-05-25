import numpy as np


class LineSearchMinimization:

    WOLFE_COND_CONST = 0.01
    BACKTRACKING_CONST = 0.5

    def __init__(self, method):
        self.method = method

    def minimize(self, f, x0, step_len, obj_tol, param_tol, max_iter):
        x = np.array(x0, dtype=float)
        x_history = [x0]
        obj_values = []

        f_prev = float("inf")
        x_prev = x.copy()

        for iteration in range(max_iter):
            f_x, g_x, h_x = f(x, True)
            obj_values.append(f_x)

            if iteration > 0:
                if np.sum(np.abs(x - x_prev)) < param_tol:
                    return x, f_x, x_history, obj_values, True
                if (f_prev - f_x) < obj_tol:
                    return x, f_x, x_history, obj_values, True

            if self.method == "Newton":
                try:
                    p = np.linalg.solve(h_x, -g_x)
                except np.linalg.LinAlgError:
                    p = -g_x
#




                # h_x_inv = np.linalg.pinv(h_x)
                # p = -np.matmul(h_x_inv, g_x)
                #
                lambda_squared = np.dot(p, np.dot(h_x, p))
                if 0.5 * lambda_squared < obj_tol:
                    return x, f_x, x_history, obj_values, True
            else:
                p = -g_x

            alpha = self.__get_step_length(f, x, p, step_len)

            x_prev = x.copy()
            f_prev = f_x
            x = x + alpha * p

            x_history.append(x.copy())

            print(f"Iteration {iteration + 1}: x = {x}, f(x) = {f_x}")

        return x, f_x, x_history, obj_values, False

    def __get_step_length(self, f, x, p, step_len):
        if step_len == "wolfe":
            return self.__wolfe(f, p, x)
        return step_len

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
