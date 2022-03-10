import numpy as np
from math import *

epsilon = 1e-4

M = np.array([[1, 2, 3], 
			[3, 4, 5], 
			[5, 4, 3],
			[0, 2, 4],
			[1, 3, 5]])

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


print(U @ sigma @ V.T)

# Set smaller singular value to 0 
sigma[1][1] = 0

# Compute the one-dimensional approximation to the matrix M.
M_approx = U @ sigma @ V.T 
print(M_approx)

# Compute new energy
new_energy = np.sum(np.square(sigma))
print(f"Energy retained = {new_energy/energy*100}%")

