import numpy as np
import matplotlib.pyplot as plt
# x = np.array([5,8,7,3])
# y = np.array([4,3,2,3])
# plt.scatter(x,y)
# u = np.linspace(3,8)
# v = 5-2*u/5
# v = 7 - 7*u/8
# v = 4 - 3*u/10
# v = 4 - 1*u/9
# plt.plot(u,v,color="red")
# plt.show()



bx = np.array([3,5])
by = np.array([6,3])

rx = np.array([1,3,3])
ry = np.array([4,1,3])

sx = np.array([4.3, 4.1, 3.6, 4.2])
sy = np.array([1.6, 1.8, 4.1, 1.9])

plt.scatter(bx,by, color="blue")
plt.scatter(rx,ry, color="red")
plt.scatter(sx, sy, color="green")
plt.show()
