from time import perf_counter
import numpy as np

from sympy import lambdify
from sympy.core.symbol import Symbol
from sympy.core.add import Add

# Optimization methods implemented in this module

TOLERANCE = 1e-6
MAX_ITER = 2000
REGULARIZATION = 1e-8  # added to the Hessian diagonal so it is always invertible


def newtons_method(
        x_symbols: tuple[Symbol],  # need to pass these for automatic differentation
        f_symbolic: Add,  # symbolic function to be minimized
        x0: tuple[float],  # initial values for each symbol
        bounds: tuple[tuple[float]],
        step_size: float = 1,  # damping factor (alpha): the full Newton step is scaled by this
        max_iter: int = MAX_ITER,
        eps: float = TOLERANCE
        ) -> tuple[list, list, float, bool]:
    '''Performs Newton's Method optimization on a symbolic function.

    Args:
        x_symbols (tuple[Symbol]): Symbols representing the variables of the function.
        f_symbolic (Add): Symbolic representation of the function to be minimized.
        x0 (np.ndarray[float]): Initial values for each symbol, given as a numpy array of floats.
        bounds (Iterable): Bounds for each variable as (min, max) tuples.
        step_size (float): Step size for the Newton updates (used for damping).
        max_iter (int, optional): Maximum number of iterations. Defaults to MAX_ITER.
        eps (float, optional): Convergence tolerance. Defaults to TOLERANCE.

    Returns:
        tuple[list, list, float, bool]: History of variable values and corresponding function values.
    '''
    assert len(x_symbols) == len(x0)
    assert step_size > 0, "Step size must be positive"
    assert step_size <= 1, "Step size must not exceed 1"

    x0 = np.array(x0).astype(float)

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

    did_converge = False
    t_start = perf_counter()
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
            did_converge = True
            break
    
    t_elapsed = perf_counter() - t_start
    return x_history, y_history, t_elapsed, did_converge


def gradient_descent(
        x_symbols: tuple[Symbol],  # need to pass these for automatic differentation
        f_symbolic: Add,  # sympy functions seem to be defined as an operation "tree", and most polynomials will have "Add" as the topmost op. TODO: make this better 
        x0: tuple[float],  # initial values for each symbol
        bounds: tuple[tuple[float]], 
        step_size: float,
        max_iter: int = MAX_ITER,
        eps: float = TOLERANCE
        ) -> tuple[list, list, float, bool]:
    '''Performs gradient descent optimization on a symbolic function.

    Args:
        x_symbols (tuple[Symbol]): Symbols representing the variables of the function.
        f_symbolic (Add): Symbolic representation of the function to be minimized.
        x0 (np.ndarray[float]): Initial values for each symbol, given as a numpy array of floats.
        bounds (Iterable): Bounds for each variable as (min, max) tuples.
        step_size (float): Step size for the gradient descent updates.
        max_iter (int, optional): Maximum number of iterations. Defaults to MAX_ITER.
        eps (float, optional): Convergence tolerance. Defaults to TOLERANCE.

    Returns:
        tuple[list, list, float, bool]: History of variable values and corresponding function values.
    '''
    assert len(x_symbols) == len(x0)
    x0 = np.array(x0).astype(float)

    # automatic differentation
    grad_f_symbolic = [f_symbolic.diff(var) for var in x_symbols]  # first-order gradient vector
    f_lambda = lambdify(x_symbols, f_symbolic, "numpy")
    grad_f_lambda = [lambdify(x_symbols, grad, "numpy") for grad in grad_f_symbolic]
    
    x_history = [x0]
    y_history = [f_lambda(*x0)]

    did_converge = False
    t_start = perf_counter()
    for i in range(max_iter):
        x_t = np.zeros_like(x0)
        for j in range(len(x0)):
            x_t[j] = x_history[i][j] - (step_size * grad_f_lambda[j](*x_history[i]))
            x_t[j] = min(max(bounds[j][0], x_t[j]), bounds[j][1])
        y_t = f_lambda(*x_t)

        x_history.append(x_t)
        y_history.append(y_t)

        if np.abs(x_t - x_history[i]).sum() < eps:  # eps is always provided
            did_converge = True
            break

    t_elapsed = perf_counter() - t_start
    return x_history, y_history, t_elapsed, did_converge


def adagrad(
        x_symbols: tuple[Symbol],  # need to pass these for automatic differentation
        f_symbolic: Add,  # sympy functions seem to be defined as an operation "tree", and most polynomials will have "Add" as the topmost op. TODO: make this better 
        x0: tuple[float],  # initial values for each symbol
        bounds: tuple[tuple[float]], 
        step_size: float,
        max_iter: int = MAX_ITER,
        eps: float = TOLERANCE
        ) -> tuple[list, list, float, bool]:
    '''Performs AdaGrad optimization on a symbolic function.

    Args:
        x_symbols (tuple[Symbol]): Symbols representing the variables of the function.
        f_symbolic (Add): Symbolic representation of the function to be minimized.
        x0 (np.ndarray[float]): Initial values for each symbol, given as a numpy array of floats.
        bounds (Iterable): Bounds for each variable as (min, max) tuples.
        step_size (float): Step size for the gradient descent updates.
        max_iter (int, optional): Maximum number of iterations. Defaults to MAX_ITER.
        eps (float, optional): Convergence tolerance. Defaults to TOLERANCE.

    Returns:
        tuple[list, list, float, bool]: History of variable values and corresponding function values.
    '''
    assert len(x_symbols) == len(x0)
    x0 = np.array(x0).astype(float)

    # automatic differentation
    grad_f_symbolic = [f_symbolic.diff(var) for var in x_symbols]  # first-order gradient vector
    f_lambda = lambdify(x_symbols, f_symbolic, "numpy")
    grad_f_lambda = [lambdify(x_symbols, grad, "numpy") for grad in grad_f_symbolic]
    
    x_history = [x0]
    y_history = [f_lambda(*x0)]

    grad_square_accum = np.zeros_like(x0)  # initial gradient accumulator matrix, empty

    did_converge = False
    t_start = perf_counter()
    for i in range(max_iter):
        # gradient accumulation step
        g = np.array([grad_f_lambda[j](*x_history[i]) for j in range(len(grad_f_lambda))])
        grad_square_accum += g**2  # accumulating outer product of the gradient vector of the most recent iteration point
        # if g =[[a,0],[0,b]] then gg^t = [[a^2,0],[0,b^2]] which is the same as g**2 for the diagonal elements
        adagrad_mult = 1.0 / (np.sqrt(grad_square_accum) + REGULARIZATION) # what about fast inverse square??

        # gradient descent step
        x_t = np.zeros_like(x0)
        for j in range(len(x0)):
            x_t[j] = x_history[i][j] - (step_size * adagrad_mult[j] * grad_f_lambda[j](*x_history[i]))
            x_t[j] = min(max(bounds[j][0], x_t[j]), bounds[j][1])
        y_t = f_lambda(*x_t)

        x_history.append(x_t)
        y_history.append(y_t)

        if np.abs(x_t - x_history[i]).sum() < eps:  # eps is always provided
            did_converge = True
            break

    t_elapsed = perf_counter() - t_start
    return x_history, y_history, t_elapsed, did_converge


def adam(x_symbols: tuple[Symbol],  # need to pass these for automatic differentation
        f_symbolic: Add,  # sympy functions seem to be defined as an operation "tree", and most polynomials will have "Add" as the topmost op. TODO: make this better 
        x0: tuple[float],  # initial values for each symbol
        bounds: tuple[tuple[float]], 
        step_size: float,
        beta1: float,
        beta2: float,
        max_iter: int = MAX_ITER,
        eps: float = TOLERANCE
        ) -> tuple[list, list, float, bool]:
    '''Performs 'adam' optimization on a symbolic function.

    Args:
        x_symbols (tuple[Symbol]): Symbols representing the variables of the function.
        f_symbolic (Add): Symbolic representation of the function to be minimized.
        x0 (np.ndarray[float]): Initial values for each symbol, given as a numpy array of floats.
        bounds (Iterable): Bounds for each variable as (min, max) tuples.
        step_size (float): Step size for the gradient descent updates.
        beta1 (float): Exponential decay rate for the first moment estimates.
        beta2 (float): Exponential decay rate for the second moment estimates.
        max_iter (int, optional): Maximum number of iterations. Defaults to MAX_ITER.
        eps (float, optional): Convergence tolerance. Defaults to TOLERANCE.

    Returns:
        tuple[list, list, float, bool]: History of variable values and corresponding function values.
    '''
    assert len(x_symbols) == len(x0), "Number of symbols must match the number of initial values"
    assert beta1 > 0 and beta1 < 1, "beta1 must be between 0 and 1"
    assert beta2 > 0 and beta2 < 1, "beta2 must be between 0 and 1"

    x0 = np.array(x0).astype(float)

    # automatic differentation
    grad_f_symbolic = [f_symbolic.diff(var) for var in x_symbols]  # first-order gradient vector
    f_lambda = lambdify(x_symbols, f_symbolic, "numpy")
    grad_f_lambda = [lambdify(x_symbols, grad, "numpy") for grad in grad_f_symbolic]
    
    x_history = [x0]
    y_history = [f_lambda(*x0)]
    m_t = np.zeros_like(x0, dtype=float)
    v_t = np.zeros_like(x0, dtype=float)

    did_converge = False
    t_start = perf_counter()
    for i in range(max_iter):
        x_t = np.zeros_like(x0)
        m_hat = np.zeros_like(x0)
        v_hat = np.zeros_like(x0)
        for j in range(len(x0)):
            gradient = grad_f_lambda[j](*x_history[i])
            m_t[j] = beta1 * m_t[j] + (1 - beta1) * gradient
            v_t[j] = beta2 * v_t[j] + (1 - beta2) * gradient ** 2

            m_hat[j] = m_t[j] / (1 - beta1 ** (i + 1))
            v_hat[j] = v_t[j] / (1 - beta2 ** (i + 1))

            x_t[j] = x_history[i][j] - step_size * m_hat[j] / (np.sqrt(v_hat[j]) + eps)
            x_t[j] = min(max(bounds[j][0], x_t[j]), bounds[j][1])

        y_t = f_lambda(*x_t)

        x_history.append(x_t)
        y_history.append(y_t)

        if np.abs(x_t - x_history[i]).sum() < eps:
            did_converge = True
            break

    t_elapsed = perf_counter() - t_start
    return x_history, y_history, t_elapsed, did_converge
