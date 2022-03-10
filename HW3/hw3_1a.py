import numpy as np
beta = 0.8
M = np.array([[1/3,1/2,0],
			[1/3,0,1/2],
			[1/3,1/2,1/2]])
v = np.array([[1/3], [1/3], [1/3]])
teleport_fac = (1-beta)*np.ones((3, 1)) / 3
n = 50
for i in range(n):
	v = beta*M@v + teleport_fac	
print(v)

N = np.array([[0, 1/2, 1, 0 ],
			  [1/3, 0, 0, 1/2],
			  [1/3, 0, 0, 1/2],
			  [1/3,1/2,0, 0 ]])
v1 = np.array([[1/4], [1/4], [1/4], [1/4]])
v2 = np.array([[1/4], [1/4], [1/4], [1/4]])
teleport_fac_A_only = (1-beta)*np.array([[1], [0], [0], [0]])
teleport_fac_A_C = (1-beta)*np.array([[1/2], [0], [1/2], [0]])

for i in range(n):
	v1 = beta*N@v1 + teleport_fac_A_only
	v2 = beta*N@v2 + teleport_fac_A_C 
print("A only: ")
print(v1)
print("A and C: ")
print(v2)

