from typing import Tuple, Iterable
import numpy as np

from sympy import lambdify
from sympy.core.symbol import Symbol
from sympy.core.add import Add

# Optimization methods implemented in this module

TOLERANCE = 1e-6
MAX_ITER = 2000
REGULARIZATION = 1e-8  # added to the Hessian diagonal so it is always invertible


def newtons_method(
        x_symbols: Tuple[Symbol],  # need to pass these for automatic differentation
        f_symbolic: Add,  # symbolic function to be minimized
        x0: np.ndarray,  # initial values for each symbol
        bounds: Iterable,
        step_size: float,  # damping factor (alpha): the full Newton step is scaled by this
        max_iter: int = MAX_ITER,
        eps: float = TOLERANCE
        ) -> Tuple[list, list]:
   
    assert len(x_symbols) == len(x0)

    n = len(x_symbols)

    # automatic differentation: first-order gradient vector and second-order Hessian matrix
    grad_f_symbolic = [f_symbolic.diff(var) for var in x_symbols]
    hess_f_symbolic = [[f_symbolic.diff(vi).diff(vj) for vj in x_symbols] for vi in x_symbols]

    f_lambda = lambdify(x_symbols, f_symbolic, "numpy")
    grad_f_lambda = [lambdify(x_symbols, g, "numpy") for g in grad_f_symbolic]
    hess_f_lambda = [[lambdify(x_symbols, h, "numpy") for h in row] for row in hess_f_symbolic]

    x_history = [x0]
    y_history = [f_lambda(*x0)]

    identity = np.eye(n)

    for i in range(max_iter):
        x_prev = x_history[i]

        # evaluate the gradient vector and Hessian matrix at the current point
        grad_val = np.array([g(*x_prev) for g in grad_f_lambda], dtype=float)
        hess_val = np.array([[h(*x_prev) for h in row] for row in hess_f_lambda], dtype=float)

        # Newton direction: solve H*d = grad (equivalent to H^{-1}*grad, but numerically safer)
        newton_step = np.linalg.solve(hess_val + REGULARIZATION * identity, grad_val)

        x_t = np.zeros_like(x0)
        for j in range(len(x0)):
            x_t[j] = x_prev[j] - (step_size * newton_step[j])
            x_t[j] = min(max(bounds[j][0], x_t[j]), bounds[j][1])
        y_t = f_lambda(*x_t)

        x_history.append(x_t)
        y_history.append(y_t)

        if np.abs(x_t - x_history[i]).sum() < eps:  # same convergence test as gradient_descent
            break

    return x_history, y_history


def gradient_descent(
        x_symbols: Tuple[Symbol],  # need to pass these for automatic differentation
        f_symbolic: Add,  # sympy functions seem to be defined as an operation "tree", and most polynomials will have "Add" as the topmost op. TODO: make this better 
        x0: np.ndarray,  # initial values for each symbol
        bounds: Iterable, 
        step_size: float,
        max_iter: int = MAX_ITER,
        eps: float = TOLERANCE
        ) -> Tuple[list, list]:
    '''Performs gradient descent optimization on a symbolic function.

    Args:
        x_symbols (Tuple[Symbol]): Symbols representing the variables of the function.
        f_symbolic (Add): Symbolic representation of the function to be minimized.
        x0 (np.ndarray[float]): Initial values for each symbol, given as a numpy array of floats (must have decimals).
        bounds (Iterable): Bounds for each variable as (min, max) tuples.
        step_size (float): Step size for the gradient descent updates.
        max_iter (int, optional): Maximum number of iterations. Defaults to MAX_ITER.
        eps (float, optional): Convergence tolerance. Defaults to TOLERANCE.

    Returns:
        Tuple[list, list]: History of variable values and corresponding function values.
    '''
    assert len(x_symbols) == len(x0)

    # automatic differentation
    grad_f_symbolic = [f_symbolic.diff(var) for var in x_symbols]  # first-order gradient vector
    f_lambda = lambdify(x_symbols, f_symbolic, "numpy")
    grad_f_lambda = [lambdify(x_symbols, grad, "numpy") for grad in grad_f_symbolic]
    
    x_history = [x0]
    y_history = [f_lambda(*x0)]

    for i in range(max_iter):
        x_t = np.zeros_like(x0)
        for j in range(len(x0)):
            x_t[j] = x_history[i][j] - (step_size * grad_f_lambda[j](*x_history[i]))
            x_t[j] = min(max(bounds[j][0], x_t[j]), bounds[j][1])
        y_t = f_lambda(*x_t)

        x_history.append(x_t)
        y_history.append(y_t)

        if np.abs(x_t - x_history[i]).sum() < eps:  # eps is always provided
            break

    return x_history, y_history


def adagrad() -> Tuple[np.ndarray, np.ndarray]:
    ...


def adam() -> Tuple[np.ndarray, np.ndarray]:
    ...