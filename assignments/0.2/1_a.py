import numpy as np
U = np.random.randint(-10,10,size=(4,4))
V = np.random.randint(0,10,size=(4,4))
b = np.ones(4)

print("Matrix U:\n%s" %U)
print("\nMatrix V:\n%s" %V)
print("\nMatrix of ones:\n%s" %b)

rank_u = np.linalg.matrix_rank(U)
rank_v = np.linalg.matrix_rank(V)
print("\nRank of U:\n%d" %rank_u)
print("\nRank of V:\n%d" %rank_v)






