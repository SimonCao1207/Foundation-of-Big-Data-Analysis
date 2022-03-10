import pyspark
from pyspark import SparkConf, SparkContext
import numpy as np
import re
import sys
import math
import csv
import os

##### Libarary used for other purpose ## 
# import itertools                     #
# import time                          #
# import matplotlib.pyplot as plt      #
########################################

def map_to_float(line):
	lst = list(map(float, line.split(' ')))
	return lst

def loadData(file):
	conf = SparkConf()
	sc = SparkContext(conf=conf)
	lines = sc.textFile(file)
	data = lines.map(map_to_float)
	return data


def distance(p, q):
	"""
	Calculate the distance between two points
	"""
	if not len(p) == len(q):
		return None
	s = []
	for i in range(len(p)):
		s.append((p[i] - q[i])**2)
	return math.sqrt(sum(s))

def initializeCentroids(points, k):
	"""
	return index of centroids
	"""
	centroids = [0]
	for _ in range(k-1):
		distances = []
		for idx, p in enumerate(points):
			d = float("inf")
			for c in centroids:
				d = min(d, distance(p, points[c]))
			distances.append(d)
		centroids.append(distances.index(max(distances)))
	return centroids

def closestPoint(point, centroids):
	"""
	return index of closest centroid.
	"""
	p, pos = point
	if pos in centroids:
		return pos
	else:
		m = float("inf")
		ans = -1
		for c, idx in centroids:
			d = distance(p, c)
			if d < m:
				m = d
				ans = idx
		return ans

def diameter(cluster):
	"""
	Calculate the diameter of the cluster (distance between two furthest points in a cluster)
	"""
	m = 0
	combinations = []
	for i in range(len(cluster)-1):
		for j in range(i+1, len(cluster)):
			combinations.append((cluster[i],cluster[j]))
	for p1, p2 in combinations:
		d = distance(p1,p2)
		if d > m:
			m = d
	return m

def addCentroid(e, dp):
	clusterId, lst = e
	lst.append(dp[clusterId])
	return lst

# start = time.time()

k = int(sys.argv[2])
data = loadData(sys.argv[1]) # Parse the data
dp = data.collect()
centroidsIdx = initializeCentroids(data.collect(), k) # indices of initialize centroids 
centroidsRDD = data.zipWithIndex().filter(lambda point: point[1] in centroidsIdx) 
centroidsLst = centroidsRDD.collect()
closest = data.zipWithIndex().map(lambda point: (closestPoint(point, centroidsLst), point[0])) # (idx, point)
closest = closest.groupByKey().mapValues(list)
diameterOfCluster = closest.map(lambda e: (e[0], diameter(e[1]))).collect()
averageDiameter = sum([d for cluster, d in diameterOfCluster]) / k
print((k, averageDiameter))


"""Plot the number-of-clusters vs. average diameter graph  with k = 1,2,4,8,16"""

# avgS = []
# lst = [1,2,4,8,16]
# for k in lst:
# 	centroidsIdx = initializeCentroids(data.collect(), k)
# 	centroidsRDD = data.zipWithIndex().filter(lambda point: point[1] in centroidsIdx)
# 	centroidsLst = centroidsRDD.collect()
# 	closest = data.zipWithIndex().map(lambda point: (closestPoint(point, centroidsLst), point[0])) # (idx, point)
# 	closest = closest.groupByKey().mapValues(list)
# 	diameterOfCluster = closest.map(lambda e: (e[0], diameter(e[1]))).collect()
# 	averageDiameter = sum([d for cluster, d in diameterOfCluster]) / k
# 	avgS.append(averageDiameter)

# for idx, avg in enumerate(avgS):
# 	print((2**idx, avg))

# plt.plot(lst, avgS)
# plt.ylabel('Average Diameter')
# plt.xlabel('Number of Clusters')
# plt.show()
# end = time.time()
# print("running time = ", end-start)