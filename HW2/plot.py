import matplotlib.pyplot as plt

k = [4,5,6,7,8]
value = [2018.2674, 1526.2138, 1420.7214, 1326.9456, 1003.3226]

plt.plot(k, value)
plt.ylabel('Average Diameter')
plt.xlabel('Number of Clusters')
plt.show()