import numpy as np
from math import *

TOLERATE = 0.00000001 # epsilon number (assume 0 if smaller than this constant)

def frobenius_norm(vec):
	return sqrt(np.sum(np.square(vec)))

def find_eigen(ma, initV):
	"""
	ma: Matrix
	initV: initial non zero vector
	"""
	x0 = initV
	x1 = (ma @ x0) / frobenius_norm(ma @ x0)
	principle_eigen = [x0, x1]
	dif = frobenius_norm(principle_eigen[-1] - principle_eigen[-2])

	while dif > TOLERATE:
		e = ma @ principle_eigen[-1]
		principle_eigen.append(e/frobenius_norm(e))
		dif = frobenius_norm(principle_eigen[-1] - principle_eigen[-2])

	estimated_x = principle_eigen[-1]   #  estimate the principal eigenvector for the matrix.
	estimated_val = (estimated_x).T @ ma @ (estimated_x) #  estimate the principal eigenvalue for the matrix.

	return estimated_x, estimated_val



M = np.array([[1, 1, 1], 
			[1, 2, 3], 
			[1, 3, 6]])

x0 = np.array([1, 1, 1])

x1, val1 = find_eigen(M, x0)

x1 = x1.reshape(-1, 1)
M2 = M - val1*(x1 @ x1.T)

x2, val2 = find_eigen(M2, x0)
x2 = x2.reshape(-1, 1)

M3 = M2 - val2*(x2 @ x2.T)
x3, val3 = find_eigen(M3, x0)
x3 = x3.reshape(-1, 1)

# t_val, t_vec = numpy.linalg.eig(M)


# PART A
print("Approximate value of the principal eigenvector: ")
print(x1)

# PART B
print("Approximate value of the principal eigenvalue: ")
print(val1)

# PART C
print("New construct matrix: ")
print(M2)

# PART D
print("Second eigen vector: ")
print(x2)
print("Second eigen value")
print(val2)

# PART E
print("Third eigen vector: ")
print(x3)	
print("Third eigen value")
print(val3)




# print("...................................")
# print("Ground Truth")
# print("True vector: ", t_vec)
# print("True value",t_val)




