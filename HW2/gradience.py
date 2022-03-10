import numpy as np
from math import *
epsilon = 0.001
# l1 = [.975, .700, -.675]
# l2 = [.485, -.485, .728]
# l3 = [.702, -.702, .117]
# l4 = [-.857, .286, .429]
# a,b,c = l4
# print(2*a+3*b+6*c)
# print(a**2+b**2+c**2)

M = np.array([[1, 2,3,4,5], 
			[2,3,2,5,3], 
			[5,5,5,3,2]])
row_mean = np.mean(M, axis=0)

M = M - row_mean[np.newaxis, :] 

col_mean = np.mean(M, axis=1)

M = M - col_mean[:, np.newaxis]

for i in range(len(M)):
	for j in range(len(M[0])):
		if abs(M[i][j]) < epsilon:
			M[i][j] = 0

print(M[0][3])
print(M)
def norm(vec):
	return np.sqrt(np.sum(vec * vec))

utility_matrix = np.array([[1,0,1,0,1,2], 
			[1,1,0,0,1,6], 
			[0,1,0,1,0,2]])
alphas = [0.5, 1, 2, 0.5]

for alpha in alphas:
	A = utility_matrix[0]
	B = utility_matrix[1]
	C = utility_matrix[2]

	nB = utility_matrix[1]*np.array([1,1,1,1,1,alpha])
	nA =  utility_matrix[0]*np.array([1,1,1,1,1,alpha])
	nC = utility_matrix[2]*np.array([1,1,1,1,1,alpha])

	# print(nA)

	cosine_distAB = np.dot(nA, nB) / (norm(nA) * norm(nB))
	# print("CHECK",np.dot(nA, nB))
	cosine_distBC = np.dot(nB, nC) / (norm(nB) * norm(nC))
	cosine_distAC = np.dot(nA, nC) / (norm(nA) * norm(nC))

	# cosine_distCA = np.dot(nC, nA) / (norm(nA) * norm(nC))
	# print(cosine_distAC == cosine_distCA)
	print("alpha = ", alpha)
	print("AB = ", cosine_distAB)
	print("BC = ", cosine_distBC)
	print("AC = ", cosine_distAC)
	# print(cosine_distBC > cosine_distAC)

print("ground true:")
for alpha in alphas:
	AB = (2 + 12*alpha**2)/sqrt(9 + 120*alpha**2 + 144*alpha**4)
	BC = (1 + 12*alpha**2)/sqrt(6 + 84*alpha**2 + 144*alpha**4)
	AC = (4*alpha**2)/sqrt(6 + 20*alpha**2 + 16*alpha**4)
	print("alpha = ", alpha)
	print("AB = ",AB)
	print("BC = ", BC)
	print("CA = ", AC)

a = np.array([3094, 3106, 3004, 3066, 2984, 3124, 3316, 3212, 3380, 3018])
x = np.mean(a)
s = [(a[i] - x)**2 for i in range(len(a))]
std = sqrt(1/(len(a)-1)*sum(s))
variace = 1/(len(a)-1)*sum(s)
print("std = ", std)
print("x = ", x )
print("variance = ", variace)

