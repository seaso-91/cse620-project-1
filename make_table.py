import numpy as np
import pandas as pd
from sympy import Symbol, cos
from methods import gradient_descent, newtons_method, adagrad, adam

x1, x2 = Symbol('x1'), Symbol('x2')
syms = (x1, x2)

FUNCTIONS = {
    "Quadratic":   x1**2 + x2**2,
    "Rosenbrock":  (1 - x1)**2 + 100*(x2 - x1**2)**2,
    "CosineBumps": x1**2 + x2**2 + 10*cos(x1) + 10*cos(x2),
}
STARTS       = [np.array([1.0, 2.0]), np.array([-3.0, 4.0])]
GD_STEPS     = [0.001, 0.01, 0.1]          
NEWTON_STEPS = [0.1, 0.5, 1.0]             
WIDE  = ((-9999, 9999), (-9999, 9999))
TIGHT = ((-10, 10), (-10, 10))             
MAX_ITER = 100000

def record(method, fname, x0, step, out):
    xh, yh, t, conv = out                  
    return dict(Function=fname, Method=method, Start=tuple(x0), Step=step,
                Iterations=len(xh) - 1,
                Final_x=round(float(xh[-1][0]), 5), Final_y=round(float(xh[-1][1]), 5),
                Final_f=float(yh[-1]), Time_ms=round(t * 1000, 3), Converged=conv)

rows = []
for fname, f in FUNCTIONS.items():
    for x0 in STARTS:
        for s in GD_STEPS:
            rows.append(record("Gradient Descent", fname, x0, s,
                        gradient_descent(syms, f, x0, WIDE, s, max_iter=MAX_ITER)))
            rows.append(record("AdaGrad", fname, x0, s,
                        adagrad(syms, f, x0, WIDE, s, max_iter=MAX_ITER)))
            rows.append(record("Adam", fname, x0, s,
                        adam(syms, f, x0, WIDE, s, 0.9, 0.999, max_iter=MAX_ITER)))
        for s in NEWTON_STEPS:
            b = TIGHT if fname == "CosineBumps" else WIDE
            rows.append(record("Newton", fname, x0, s,
                        newtons_method(syms, f, x0, b, s, max_iter=MAX_ITER)))

df = pd.DataFrame(rows)
df.to_csv("results_all_methods.csv", index=False)
print(df.to_string(index=False))
print(f"\nSaved results_all_methods.csv ({len(df)} rows)")