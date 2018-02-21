import sympy as sy
import numpy as np

# TASK 2 Linear systems: Use Gauss-Jordan reduction to solve the following system of linear equations:
#
#       x_1  + 3x_2 + x_3  + x_4  = 3
#       2x_1 - 2x_2 + x_3  + 2x_4 = 8
#       3x_1 + x_2  + 2x_3 - x_4  = -1
print "TASK 2\n-------------"
A = np.array([[1,3,1,1],
             [2,-2,1,2],
             [3,1,2,-1]])
b = np.array([[3],[8],[-1]])

print np.linalg.matrix_rank(A)
AA=np.column_stack([A,b])
print sy.Matrix(AA).rref()

# Task 3 Linear combinations: Consider the system of linear equations Ax = b specified by:
#
#   | 1 |       | 0 |       | 1 |       | 2 |       | 3 |
#   | 2 | x_1 + | 2 | x_2 + | 4 | x_3 + | 0 | x_4 = | 4 |
#   | 3 |       | 1 |       | 2 |       | 0 |       | 2 |
#
#   In the expression, we wrote explictly the column vectors a1, a2, a3, a4 of the matrix A.
#   Which (if any) of the following combinations of column vectors of A form a basis of the vectors
#   space that contains the vector b? For each set of vectors, if it does form a basis determine
#   the coordinates of b with respect to that basis.
#
#   (i)      {a_1, a_2, a_3}
#   (ii)     {a_1, a_3, a_4}
#   (iii)    {a_2, a_3, a_4}
#   (iv)     {a_1, a_2, a_3, a_4}

print "\n\nTASK 3\n-------------"
A = np.array([[1,0,1,2],
              [2,2,4,0],
              [3,1,2,0]])
b = np.array([[3],[4],[2]])
print np.linalg.det(A[:,[0,1,2]])
print np.linalg.det(A[:,[0,2,3]])
print np.linalg.det(A[:,[1,2,3]])
print np.linalg.solve(A[:,[0,1,2]],b)
print np.linalg.solve(A[:,[0,2,3]],b)
