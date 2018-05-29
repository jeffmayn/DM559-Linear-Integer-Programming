import sympy as sy
import numpy as np
import math as math
from fractions import Fraction as f
np.set_printoptions(precision=3,suppress=True)

def tofrac(A):
    return np.array([map(lambda x: f(str(x)), S) for S in A])

A = np.array([[1, 3, 1],
              [2,-2, 1],
              [3, 1, 2]])

B = np.array([[1, 3, 1],
              [2,-2, 1],
              [3, 1, 2]])

SOLVE = tofrac(np.linalg.solve(A,B))
print(SOLVE)
