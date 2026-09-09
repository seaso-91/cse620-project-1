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

print(results)
print(f'Final values: {results[0][-1]}, found minimum: {results[1][-1]}')