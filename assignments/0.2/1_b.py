import numpy as np
U = np.random.randint(-10,10,size=(4,4))
V = np.random.randint(0,10,size=(4,4))
C = np.random.randint(0,10,size=(4,1))
b = np.ones(4)

print(V)
print(C)

u1, u2, u3, u4, c = ([] for i in range(5))
n = 0
while n < len(U):
	u1.append(U[n,0])
	u2.append(U[n,1])
	u3.append(U[n,2])
	u4.append(U[n,3])
	c.append(C[n,0])
	n+=1

Un_dot_Cn = np.dot(u1,C[0,0]) + \
	    np.dot(u2,C[1,0]) + \
	    np.dot(u3,C[2,0]) + \
	    np.dot(u4,C[3,0])

U_dot_c = np.dot(U,c)

print("Vector b corrosponds to original?: %s" \
	%np.allclose(Un_dot_Cn, U_dot_c, rtol=1e-05, atol=1e-08))
