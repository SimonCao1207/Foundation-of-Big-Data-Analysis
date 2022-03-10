import pyspark
from pyspark import SparkConf, SparkContext
import numpy as np
import re
import sys
import math
import csv
import os

# CANNOT USE NUMPY -> TO DO

def loadData(file):
	"""
	Return a list of tuple pairs in (start point -> end point)
	"""
	conf = SparkConf()
	sc = SparkContext(conf=conf)
	lines = sc.textFile(file)
	data = lines.map(lambda line: tuple(map(int, line.split('	')))).distinct()
	return data

def contrib(in_nodes, out_degree, n):
	# r = np.zeros((1,n)) # initialize row of transition matrix
	r = [0 for _ in range(n)]
	for node in in_nodes:
		# r[0][node-1] = 1 / out_degree[node]
		r[node-1] = 1 / out_degree[node]
	return r

def dot_product(v1, v2):
	if len(v1) != len(v2):
		return None
	s = 0
	for i in range(len(v1)):
		s += v1[i]*v2[i] 
	return s

def add_vec(v1, v2):
	if len(v1) != len(v2):
		return None
	res = [v1[i] + v2[i] for i in range(len(v1))]
	return res

data = loadData(sys.argv[1]) # Load data
num_nodes = data.flatMap(lambda p: p).distinct().count() # number of nodes (default is 1000)
out_degree = data.countByKey() # dictionary of out-degree of each node
reverse_pair = data.map(lambda p: tuple(reversed(p))) # Reverse into (destination , starting point)
reverse_group = reverse_pair.groupByKey() # each node is group to all the points heading to it.

# print(reverse_group.mapValues(list).collect())

#Transition matrix M
M = reverse_group.map(lambda x : (x[0], contrib(x[1], out_degree, num_nodes)))

# Initial pageRank vector
# r = np.ones((num_nodes, 1)) / num_nodes
r = [1/num_nodes for _ in range(num_nodes)]

# Parameters
beta = 0.9
num_iter = 50

# teleport factor
teleport_fac = (1-beta)/num_nodes

for _ in range(num_iter):
	# rdd = M.map(lambda x: (x[0], (x[1] @ r * beta)[0][0]))
	rdd = M.map(lambda x: (x[0], (dot_product(x[1], r) * beta)))
	# new_r = np.zeros((num_nodes, 1))
	new_r = [0 for _ in range(num_nodes)]

	for (node, val) in rdd.collect():
		# new_r[node-1][0] += val
		# new_r[node-1][0] += teleport_fac
		new_r[node-1] += val
		new_r[node-1] += teleport_fac
	r = new_r
# print(r)
# lst_r = r[:,0]
# top_10_idx = np.argsort(lst_r)[-10:] # Top 10 page ID
top_10_idx = sorted(range(len(r)), key=lambda i: r[i])[-10:] # Top 10 page ID
top_10_score = [r[i] for i in top_10_idx] # Top 10 scores

for i in range(len(top_10_idx)-1,-1,-1):
	print(f"{top_10_idx[i]+1}\t{top_10_score[i]:.5f}")




