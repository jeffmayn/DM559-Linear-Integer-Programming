import sympy as sy
import numpy as np

A = np.array([[1,3,1,1],
            [2,-2,1,2],
            [3,1,2,-1]])
b = np.array([[3],[8],[-1]])

print ("")
print ("A =")
print(A)

print ("\n")
print ("Rank =")
print (np.linalg.matrix_rank(A))

AA=np.column_stack([A,b])
print ("")
print (sy.Matrix(AA).rref())
print()
