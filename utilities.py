import numpy as np
import matplotlib.pyplot as plt
from sympy import lambdify
from sympy.core.symbol import Symbol
from sympy.core.add import Add
from typing import Tuple

def plot_contours_with_path(
    x_symbols: Tuple[Symbol],  # need to pass these for automatic differentation
    f_symbolic: Add,  # sympy functions seem to be defined as an operation "tree", and most polynomials will have "Add" as the topmost op. TODO: make this better 
    results:   Tuple[list, list],
    xlim:   tuple[int, int] = None,
    ylim:   tuple[int, int] = None,
    levels: int = 40,
    title:  str = ""
    ):

    P = np.array(results[0][:])

    if xlim is None:
        xlim = (min(P[:,0])-abs(min(P[:,0]))*0.25, max(P[:,0])+abs(max(P[:,0]))*0.25)
    if ylim is None:
        ylim = (min(P[:,1])-abs(min(P[:,1]))*0.25, max(P[:,1])+abs(max(P[:,1]))*0.25)

    f_lambda = lambdify(x_symbols, f_symbolic)
    xs = np.linspace(xlim[0], xlim[1], 400)
    ys = np.linspace(ylim[0], ylim[1], 400)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f_lambda(X[i, j], Y[i, j])
    plt.figure()
    cs = plt.contour(X, Y, Z, levels=levels)
    plt.clabel(cs, inline=1, fontsize=8)
    plt.plot(P[:,0], P[:,1], marker='o', linewidth=1)
    plt.title(title)
    plt.xlabel('x'); plt.ylabel('y')
    plt.show()

