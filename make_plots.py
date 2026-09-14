"""make_plots.py - generate figures from the team's optimizers in methods.py.
Run:  python3 make_plots.py   ->  writes landscape_f1/f2/f3.png and trajectories_all.png
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from sympy import Symbol, cos, lambdify
from methods import gradient_descent, newtons_method, adagrad, adam

x1, x2 = Symbol('x1'), Symbol('x2')
SYMS = (x1, x2)

FUNCS = {
    "f1": dict(title="$f_1$: Convex bowl", expr=x1**2 + x2**2,
               dom=(-4, 4, -4, 4), start=np.array([-3.0, 4.0]), tmin=(0, 0)),
    "f2": dict(title="$f_2$: Rosenbrock", expr=(1 - x1)**2 + 100*(x2 - x1**2)**2,
               dom=(-2, 2, -1, 3), start=np.array([1.0, 2.0]), tmin=(1, 1)),
    "f3": dict(title="$f_3$: Cosine bumps", expr=x1**2 + x2**2 + 10*cos(x1) + 10*cos(x2),
               dom=(-6, 6, -6, 6), start=np.array([-3.0, 4.0]), tmin=None),
}
PLOT_ITERS = 8000   # iterations to draw (keeps figures clean/fast)

STYLE = {"Gradient Descent": "#1f77b4", "Newton": "#d62728",
         "AdaGrad": "#2ca02c", "Adam": "#9467bd"}

def grid(expr, dom, n=350):
    f = lambdify(SYMS, expr, "numpy")
    X, Y = np.meshgrid(np.linspace(dom[0], dom[1], n), np.linspace(dom[2], dom[3], n))
    return X, Y, f(X, Y)

def sub(path, k=600):
    if len(path) <= k: return path
    idx = np.unique(np.r_[np.linspace(0, len(path)-1, k).astype(int), len(path)-1])
    return path[idx]

def run(fkey):
    fd = FUNCS[fkey]; expr, start = fd["expr"], fd["start"]
    wide = ((-9999, 9999), (-9999, 9999)); tight = ((-10, 10), (-10, 10))
    nb = tight if fkey == "f3" else wide
    out = {}
    out["Gradient Descent"] = gradient_descent(SYMS, expr, start, wide,
                                0.001 if fkey == "f2" else 0.1, max_iter=PLOT_ITERS)[0]
    out["AdaGrad"] = adagrad(SYMS, expr, start, wide, 0.1, max_iter=PLOT_ITERS)[0]
    out["Adam"]    = adam(SYMS, expr, start, wide, 0.1, 0.9, 0.999, max_iter=PLOT_ITERS)[0]
    out["Newton"]  = newtons_method(SYMS, expr, start, nb,
                                0.5 if fkey in ("f2", "f3") else 1.0, max_iter=PLOT_ITERS)[0]
    return {m: np.array(h) for m, h in out.items()}

# 1) landscape contour maps
for k, fd in FUNCS.items():
    X, Y, Z = grid(fd["expr"], fd["dom"])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    cs = ax.contourf(X, Y, Z, levels=40, cmap="viridis")
    ax.contour(X, Y, Z, levels=40, colors="k", linewidths=0.25, alpha=0.35)
    fig.colorbar(cs, ax=ax, label="f(x, y)")
    if fd["tmin"]: ax.plot(*fd["tmin"], "r*", ms=15, label="global min"); ax.legend(loc="upper right")
    ax.set(xlabel="x", ylabel="y", title=fd["title"])
    fig.tight_layout(); fig.savefig(f"landscape_{k}.png", dpi=130); plt.close(fig)

# 2) combined trajectory comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (k, fd) in zip(axes, FUNCS.items()):
    X, Y, Z = grid(fd["expr"], fd["dom"])
    ax.contour(X, Y, Z, levels=30, cmap="viridis", linewidths=0.6, alpha=0.6)
    for m, p in run(k).items():
        ps = sub(p)
        ax.plot(ps[:, 0], ps[:, 1], "-", color=STYLE[m], lw=1.8, label=m, alpha=0.9)
        ax.plot(p[-1, 0], p[-1, 1], "o", color=STYLE[m], ms=7, mec="k", mew=0.5)
    ax.plot(*fd["start"], "ks", ms=9, label="start")
    if fd["tmin"]: ax.plot(*fd["tmin"], "r*", ms=15)
    ax.set(xlim=fd["dom"][:2], ylim=fd["dom"][2:], xlabel="x", ylabel="y", title=fd["title"])
    ax.legend(fontsize=8, loc="best")
fig.suptitle("Optimizer trajectories: Gradient Descent vs Newton vs AdaGrad vs Adam", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("trajectories_all.png", dpi=130); plt.close(fig)
print("saved: landscape_f1.png, landscape_f2.png, landscape_f3.png, trajectories_all.png")
