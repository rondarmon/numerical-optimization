import numpy as np


def circles(x, need_hessian=False):
    Q = np.array([[1, 0], [0, 1]])
    f = 0.5 * (x @ Q @ x)
    g = Q @ x
    h = Q if need_hessian else None
    return f, g, h


def ellipses(x, need_hessian=False):
    Q = np.array([[1, 0], [0, 100]])
    f = 0.5 * np.matmul(np.matmul(x.transpose(), Q), x)
    g = np.matmul(Q, x)
    h = Q if need_hessian else None
    return f, g, h


def rotated_ellipses(x, need_hessian=False):
    Q = np.array([[100, 0], [0, 1]])
    R = np.array([[np.sqrt(3)/2, -0.5], [0.5, np.sqrt(3)/2]])
    Q = np.matmul(np.matmul(R.transpose(), Q), R)
    f = 0.5 * np.matmul(np.matmul(x.transpose(), Q), x)
    g = np.matmul(Q, x)
    h = Q if need_hessian else None
    return f, g, h


def rosenbrock(x, need_hessian=False):
    f = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    g = np.array([
        2 * (200 * x[0] ** 3 - 200 * x[0] * x[1] + x[0] - 1)
        , 200 * (x[1] - x[0] ** 2),
    ])
    h = np.array([
        [1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]],
        [-400 * x[0], 200],
    ]) if need_hessian else None
    return f, g, h


def linear(x, need_hessian=False):
    a = np.array([1, 2])
    f = np.matmul(a.transpose(), x)
    g = a
    h = np.zeros((2, 2)) if need_hessian else None
    return f, g, h


def triangles(x, need_hessian=False):
    first_term = x[0] + 3 * x[1] - 0.1
    second_term = x[0] - 3 * x[1] - 0.1
    third_term = -x[0] - 0.1

    exp1 = np.exp(first_term)
    exp2 = np.exp(second_term)
    exp3 = np.exp(third_term)

    f = exp1 + exp2 + exp3

    g = np.array([
        exp1 + exp2 - exp3,
        3 * (exp1 - exp2),
    ])

    h = np.array([
            [exp1 + exp2 + exp3, 0],
            [0, 9 * (exp1 + exp2)]
        ]) if need_hessian else None

    return f, g, h


def qp(x, need_hessian=False):
    f = x[0] ** 2 + x[1] ** 2 + (x[2] + 1) ** 2
    g = np.array([2 * x[0], 2 * x[1], 2 * x[2] + 2]).transpose()
    h = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]]) if need_hessian else None
    return f, g, h


def qp_ineq_constraint_1(x, need_hessian=False):
    f = -x[0]
    g = np.array([-1, 0, 0]).transpose()
    h = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]) if need_hessian else None
    return f, g, h


def qp_ineq_constraint_2(x, need_hessian=False):
    f = -x[1]
    g = np.array([0, -1, 0]).transpose()
    h = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]) if need_hessian else None
    return f, g, h


def qp_ineq_constraint_3(x, need_hessian=False):
    f = -x[2]
    g = np.array([0, 0, -1]).transpose()
    h = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]) if need_hessian else None
    return f, g, h


def lp(x, need_hessian=False):
    f = -x[0] - x[1]
    g = np.array([-1, -1]).transpose()
    h = np.array([[0, 0], [0, 0]]) if need_hessian else None
    return f, g, h


def lp_ineq_constraint_1(x, need_hessian=False):
    f = -x[0] - x[1] + 1
    g = np.array([-1, -1]).transpose()
    h = np.array([[0, 0], [0, 0]]) if need_hessian else None
    return f, g, h


def lp_ineq_constraint_2(x, need_hessian=False):
    f = x[1] - 1
    g = np.array([0, 1]).transpose()
    h = np.array([[0, 0], [0, 0]]) if need_hessian else None
    return f, g, h


def lp_ineq_constraint_3(x, need_hessian=False):
    f = x[0] - 2
    g = np.array([1, 0]).transpose()
    h = np.array([[0, 0], [0, 0]]) if need_hessian else None
    return f, g, h

def lp_ineq_constraint_4(x, need_hessian=False):
    f = -x[1]
    g = np.array([0, -1]).transpose()
    h = np.array([[0, 0], [0, 0]]) if need_hessian else None
    return f, g, h