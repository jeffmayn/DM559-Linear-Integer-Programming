import numpy as np

U = np.random.randint(-10,10,size=(4,4))
V = np.random.randint(0,10,size=(4,4))
b = np.ones(4)

# transition matrices
T = np.linalg.solve(U,V) 
S = np.linalg.solve(V,U)

# Coordinate vectors 
C = np.dot(T,b) 
D = np.dot(S,b)

#First the vectors are changed to their transponding vector in the other basis. 
VeS = np.dot(S,C) 
VeT = np.dot(T,D)

# The changed vectors are changed back to their original basis 
NewC = np.dot(T, VeS) 
NewD = np.dot(S, VeT)

#verification using allclose that C and NewC are the same. 
print(np.allclose(C,NewC, rtol=1e-05, atol=1e-08)) 
#verification using allclose that D and NewD are the same. 
print(np.allclose(D,NewD, rtol=1e-05, atol=1e-08))

