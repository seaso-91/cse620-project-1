import numpy as np
import matplotlib.pyplot as plt
from sympy import lambdify
from sympy.core.symbol import Symbol
from sympy.core.add import Add
from typing import Tuple

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] 

def plot_contours_with_path(
    x_symbols: Tuple[Symbol],  # need to pass these for automatic differentation
    f_symbolic: Add,  # sympy functions seem to be defined as an operation "tree", and most polynomials will have "Add" as the topmost op. TODO: make this better 
    results:   Tuple[list, list] | list[Tuple[list, list]],
    result_labels: list[str] | str = "path",
    xlim:   tuple[int, int] = None,
    ylim:   tuple[int, int] = None,
    levels: int = 40,
    title:  str = ""
    ):
    
    # convert results to a list to allow any number of paths to be plotted
    if isinstance(results, Tuple):
        P = [np.array(results[0][:])]
    elif isinstance(results, list):
        P=[]
        for result in results:
            P.append(np.array(result[0][:]))
    else:
        raise ValueError("Unsupported type for results")

    if isinstance(result_labels, str):
        result_labels = [result_labels]
    elif isinstance(result_labels, list):
        pass
    else:
        raise ValueError("Unsupported type for result_labels")

    if len(result_labels) != len(P):
        raise ValueError("The number of result labels must match the number of paths")

    if xlim is None:
        min_x = min([p[:,0].min() for p in P])
        max_x = max([p[:,0].max() for p in P])
        range_x = max_x - min_x
        xlim = (min_x-range_x*0.1, max_x+range_x*0.1)
    if ylim is None:
        min_y = min([p[:,1].min() for p in P])
        max_y = max([p[:,1].max() for p in P])
        range_y = max_y - min_y
        ylim = (min_y-range_y*0.1, max_y+range_y*0.1)

    f_lambda = lambdify(x_symbols, f_symbolic)
    xs = np.linspace(xlim[0], xlim[1], 400)
    ys = np.linspace(ylim[0], ylim[1], 400)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f_lambda(X[i, j], Y[i, j])
    plt.figure()

    import matplotlib.colors as mcolors
    gray_cmap = mcolors.LinearSegmentedColormap.from_list(
        "gray_no_white", ["lightgray", "black"]
    )

    cs = plt.contour(X, Y, Z, levels=levels, cmap=gray_cmap, linewidths=0.5)
    plt.clabel(cs, inline=1, fontsize=8)

    for i, path in enumerate(P):
        plt.plot(path[:,0], path[:,1], label=result_labels[i], marker='.', markersize=1, linewidth=1, color=colors[i % len(colors)])
    plt.title(title)
    plt.xlabel('x'); plt.ylabel('y')

    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
    plt.tight_layout()

    return plt

def test_optimizer(optimizer_name: str, step_size: list[float], starting_point: list, **kwargs):
    final_value =[]
    number_of_iterations = []
    starting_points = []
    time_elapsed_list = []
    convergence_list = []

    match optimizer_name:
        case "adam":
            from methods import adam as optimizer
        case "gradient_descent":
            from methods import gradient_descent as optimizer
        case "adagrad":
            from methods import adagrad as optimizer
        case "newtons_method":
            from methods import newtons_method as optimizer
        case _:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    for s in step_size:
        x_history, y_history, time_elapsed, convergence = optimizer(x0=starting_point, step_size=s, **kwargs)
        starting_points.append(starting_point)
        final_value.append(y_history[-1])
        number_of_iterations.append(len(x_history) - 1)  # -1 since x_history includes x0
        time_elapsed_list.append(time_elapsed)
        convergence_list.append(convergence)

    return step_size, starting_points, final_value, number_of_iterations, time_elapsed_list, convergence_list