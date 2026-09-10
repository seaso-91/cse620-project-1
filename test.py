# Testing script, not necessary to transfer to main branch unless desired

import numpy as np
from sympy import symbols
from methods import gradient_descent


x1, x2 = symbols('x1 x2')
func = 2*x1**2 + 3*x2**2 - 12  # minimum should be -12 at x1=0, x2=0)

results = gradient_descent((x1, x2), 
                           func, 
                           np.array([5.0, 5.0]), # these NEED to be given decimals, or else the grad descent treats them as ints
                           ((-10, 10), (-10, 10)),
                           0.1)

<<<<<<< HEAD
print(results)
print(f'Final values: {results[0][-1]}, found minimum: {results[1][-1]}')

# --- Newton's Method tests (added for local verification) ---
from methods import newtons_method
from sympy import cos

# 1) Convex bowl f1 = x1^2 + x2^2, min 0 at (0,0): Newton should finish in ~1 step
nx, ny = newtons_method((x1, x2),
                        x1**2 + x2**2,
                        np.array([3.0, 3.0]),
                        ((-10, 10), (-10, 10)),
                        1.0)  # step_size = damping factor alpha
print(f"[Newton f1 bowl]  final: {nx[-1]}, min: {ny[-1]}, iters: {len(nx)-1}  (expect ~(0,0), 0)")

# 2) Rosenbrock f2, min 0 at (1,1)
rx, ry = newtons_method((x1, x2),
                        (1 - x1)**2 + 100*(x2 - x1**2)**2,
                        np.array([-2.0, 2.0]),
                        ((-5, 5), (-5, 5)),
                        1.0)
print(f"[Newton f2 rosen] final: {rx[-1]}, min: {ry[-1]:.2e}, iters: {len(rx)-1}  (expect ~(1,1), ~0)")

# 3) Cosine-bumps f3 = x^2+y^2+10cos x+10cos y (multimodal); damped step alpha=0.5
cx, cy = newtons_method((x1, x2),
                        x1**2 + x2**2 + 10*cos(x1) + 10*cos(x2),
                        np.array([1.0, 1.0]),
                        ((-6, 6), (-6, 6)),
                        0.5)
print(f"[Newton f3 bumps] final: {cx[-1]}, min: {cy[-1]:.4f}, iters: {len(cx)-1}  (converges to a stationary point)")
=======
# print(results)
# print(f'Final values: {results[0][-1]}, found minimum: {results[1][-1]}')
from utilities import plot_contours_with_path
plot_contours_with_path((x1, x2), func, results, title="Gradient Descent Optimization Path")
>>>>>>> origin/main
