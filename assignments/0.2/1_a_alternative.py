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

u1 = []
u2 = []
u3 = []
u4 = []
c = []
for x in range (0, len(U)):
	u1.append(U[x][0])
	u2.append(U[x][1])
	u3.append(U[x][2])
	u4.append(U[x][3])
	c.append(V[x][0])

print(u1)

product1 = np.dot(u1,c[0]) + np.dot(u2,c[1]) + np.dot(u3,c[2]) + np.dot(u4,c[3])
product2 = np.dot(U,c)
print("\nDoes computed vector b corrospond to original:")
print(np.allclose(product1, product2))
