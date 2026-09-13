"""
run_experiments.py
Runs ALL FOUR optimizers (Gradient Descent, Newton, AdaGrad, Adam) across all
three functions, multiple starting points, and multiple step sizes, then
collects every run's results (iterations, final point, final value, time to
converge, and whether it converged) into one table for the report.

Grid: 4 methods x 3 step sizes x 2 starting points x 3 functions = 72 runs.
"""
import warnings
import numpy as np
import pandas as pd
from sympy import Symbol, cos
from methods import gradient_descent, newtons_method, adagrad, adam

warnings.filterwarnings("ignore")  # silence overflow warnings from diverging runs

# ---- the three functions ----
x1, x2 = Symbol('x1'), Symbol('x2')
SYMS = (x1, x2)
FUNCTIONS = {
    "Quadratic":   x1**2 + x2**2,
    "Rosenbrock":  (1 - x1)**2 + 100*(x2 - x1**2)**2,
    "CosineBumps": x1**2 + x2**2 + 10*cos(x1) + 10*cos(x2),
}

# ---- experiment settings (edit these to match what your team agrees on) ----
STARTING_POINTS = [np.array([1.0, 2.0]), np.array([-3.0, 4.0])]  # 2 per the assessment
GD_STEP_SIZES     = [0.001, 0.01, 0.1]   # for GD / AdaGrad / Adam
NEWTON_STEP_SIZES = [0.1, 0.5, 1.0]      # Newton's damping factors (must be <= 1)
MAX_ITER = 100000   # iteration cap for every optimizer
WIDE  = ((-9999, 9999), (-9999, 9999))
TIGHT = ((-10, 10), (-10, 10))           # Newton needs tight bounds on CosineBumps

def _record(method, fname, x0, step, out):
    x_hist, y_hist, t_elapsed, converged = out
    xf = x_hist[-1]
    return {
        "Function": fname, "Method": method,
        "Start": f"({x0[0]:g}, {x0[1]:g})", "Step": step,
        "Iterations": len(x_hist) - 1,
        "Final x": round(float(xf[0]), 5), "Final y": round(float(xf[1]), 5),
        "Final f": float(y_hist[-1]),
        "Time (ms)": round(t_elapsed * 1e3, 3),
        "Converged": converged,
    }

def run_all():
    rows = []
    for fname, expr in FUNCTIONS.items():
        for x0 in STARTING_POINTS:
            for s in GD_STEP_SIZES:
                rows.append(_record("Gradient Descent", fname, x0, s,
                            gradient_descent(SYMS, expr, x0, WIDE, s, max_iter=MAX_ITER)))
                rows.append(_record("AdaGrad", fname, x0, s,
                            adagrad(SYMS, expr, x0, WIDE, s, max_iter=MAX_ITER)))
                rows.append(_record("Adam", fname, x0, s,
                            adam(SYMS, expr, x0, WIDE, s, 0.9, 0.999, max_iter=MAX_ITER)))
            for s in NEWTON_STEP_SIZES:
                bounds = TIGHT if fname == "CosineBumps" else WIDE
                rows.append(_record("Newton", fname, x0, s,
                            newtons_method(SYMS, expr, x0, bounds, s, max_iter=MAX_ITER)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run_all()
    df.to_csv("results_all_methods.csv", index=False)
    print(f"Ran {len(df)} experiments -> saved results_all_methods.csv\n")
    # compact preview: best (lowest final f) run per Function x Method
    pd.set_option("display.width", 200, "display.max_columns", 20)
    best = (df.sort_values("Final f")
              .groupby(["Function", "Method"], as_index=False).first()
              .sort_values(["Function", "Method"]))
    print("Best run per Function x Method (lowest final f):")
    print(best[["Function","Method","Step","Iterations","Final f","Time (ms)","Converged"]].to_string(index=False))
