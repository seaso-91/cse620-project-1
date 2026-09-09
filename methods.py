from typing import Tuple, Iterable
import numpy as np

from sympy import lambdify
from sympy.core.symbol import Symbol
from sympy.core.add import Add

# Optimization methods implemented in this module

TOLERANCE = 1e-6
MAX_ITER = 2000


def newtons_method() -> Tuple[np.ndarray, np.ndarray]:
    ...


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
        x0 (np.ndarray): Initial values for each symbol.
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