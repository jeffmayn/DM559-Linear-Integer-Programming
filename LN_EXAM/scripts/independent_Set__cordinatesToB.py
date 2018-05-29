# Used to find linear independent vectors of R, with respect to b.
# Then prints the coefficients to form b ith the independent set.

import numpy as np
import sympy as sy
A = np.array([[1,0,1,2],
[2,2,4,0],
[3,1,2,0]])
b = np.array([[3],[4],[2]])

# Choosen vectors must form a square
print (np.linalg.det(A[:,[0,1,2]]))
print (np.linalg.det(A[:,[0,2,3]]))
print (np.linalg.det(A[:,[1,2,3]]))

# If the det != 0, they are independent.

# Then calculate coefficients , with respect to the given vector set by:

print (np.linalg.solve(A[:,[0,1,2]],b))


input("Press Enter to continue...") # Python 3
# raw_input("Press Enter to continue...") # Python 2
