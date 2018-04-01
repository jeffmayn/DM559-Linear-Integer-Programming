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

C = np.random.randint(0,10,size=(4,1))

u1 = np.matrix([ [U[0,0]],
		 [U[1,0]],
		 [U[2,0]],
		 [U[3,0]]])

u2 = np.matrix([ [U[0,1]],
		 [U[1,1]],
		 [U[2,1]],
		 [U[3,1]]])

u3 = np.matrix([ [U[0,2]],
		 [U[1,2]],
		 [U[2,2]],
		 [U[3,2]]])

u4 = np.matrix([ [U[0,3]],
		 [U[1,3]],
		 [U[2,3]],
		 [U[3,3]]])

product1 = np.dot(u1,C[0,0]) + np.dot(u2,C[1,0]) + np.dot(u3,C[2,0]) + np.dot(u4,C[3,0])
product2 = np.dot(U,C)
print("\nDoes computed vector b corrospond to original:")
print(np.allclose(product1, product2))

#print("\nC_1:\n%s" %u1)
#print("\nC_2:\n%s" %u2)
#print("\nC_3:\n%s" %u3)
#print("\nC_4:\n%s" %u4)





