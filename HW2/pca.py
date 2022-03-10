import numpy as np
from math import *

epsilon = 1e-4

M = np.array([[1, 2], 
			[2, 2], 
			[3, 4]])

# Compute the matrices M.T @ M and M @ M.T
print("M x M.T = ")
print(M @ M.T)
print("M.T x M = ")
print(M.T @ M)

# Find the eigenpairs (eigenvalues, eigenvectors) for your matrices of part 
eigenValV, V = np.linalg.eig(M.T @ M)
eigenValU, U = np.linalg.eig(M @ M.T)
print(f"Eigenvector of M.T x M : ")
print(V)
print(f"Eigenvalue of M.T x M : {eigenValV}")
print(f"Eigenvector of M x M.T : ")
print(U)
print(f"Eigenvalue of M x M.T : {eigenValU}")

# Find the SVD for the original matrix M from part b
delIdxU = []
delIdxV = []
for idx, val in enumerate(eigenValU):
	if abs(val) < epsilon:
		delIdxU.append(idx)

for idx, val in enumerate(eigenValV):
	if abs(val) < epsilon:
		delIdxV.append(idx)

eigenValU = np.delete(eigenValU, delIdxU)
U = np.delete(U, delIdxU, axis=1)

eigenValV = np.delete(eigenValV, delIdxV)
V = -np.delete(V, delIdxV, axis=1)

energy = sum(eigenValV)
sigma = np.sqrt(np.diag(eigenValV))

print("U = ")
print(U)
print("sigma = ")
print(sigma)
print("V = ")
print(V)


print("Reconstructed matrix",U @ sigma @ V.T)

# Set smaller singular value to 0 
sigma[1][1] = 0

# Compute the one-dimensional approximation to the matrix M.
M_approx = U @ sigma @ V.T 
print(M_approx)

# Compute new energy
new_energy = np.sum(np.square(sigma))
print(f"Energy retained = {new_energy/energy*100}%")



M = np.array([[0, 0, 1], 
			[1/2, 0, 0], 
			[1/2, 1, 0]])
v = np.array([[1],[1],[1]])

beta = 0.85

print(M@M@M@M@M@v)



for i in  range(4):
	x = float(input())
	k = int(input())
	I_O = 4*(21 + k + 3*(x + (1-x)*k))
	print("result = ", I_O)