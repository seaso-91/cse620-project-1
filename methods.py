from typing import Tuple, Iterable
import numpy as np

# Optimization methods implemented in this module

TOLERANCE = 1e-6
MAX_ITER = 2000


def newtons_method() -> Tuple[np.ndarray, np.ndarray]:
    ...

def gradient_descent(
        f_x: callable, 
        df_dx: callable, 
        x0: float, 
        bounds: Iterable, 
        step_size: float,
        max_iter: int = MAX_ITER,
        eps: float = TOLERANCE
        ) -> Tuple[np.ndarray, np.ndarray]:
    
    x_history = [x0]
    y_history = [f_x(x0)]

    for i in range(0, max_iter):
        x_t = x_history[i] - (step_size * df_dx(x_history[i]))
        x_t = min(max(bounds[0], x_t), bounds[1]) # Ensures that no movement occurs outside of the function bounds
        y_t = f_x(x_t)

        x_history.append(x_t)
        y_history.append(y_t)

        if eps:
            if np.abs(x_t - x_history[i]) < eps:
                break

    return x_history, y_history

def adagrad() -> Tuple[np.ndarray, np.ndarray]:
    ...

def adam() -> Tuple[np.ndarray, np.ndarray]:
    ...