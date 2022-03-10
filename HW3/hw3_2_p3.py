import re
import sys
from math import *
import csv
import time
import collections
import itertools
import os
import numpy as np

def loadData(file):
    edges = []
    deg = {}
    nodes = set()
    edge_idx = {}
    with open(file) as f:
        for line in f:
            a,b,_ = line.split("\t")
            a,b = int(a), int(b)
            edges.append((a,b))
            nodes.add(int(a))
            nodes.add(int(b))
        edge_unique = set()
        for edge in edges:
            if edge[0] > edge[1]:
                edge_unique.add((edge[1], edge[0]))
            else:
                edge_unique.add(edge)
        
        # edge_idx = {node: {node : 0 for node in nodes} for node in nodes}
        
        for edge in edge_unique:
            a,b = edge
            if not a in deg:
                deg[a] = 0
            else:
                deg[a] += 1
            if not b in deg:
                deg[b] = 0
            else:
                deg[b] += 1

            if not a in edge_idx:
                edge_idx[a] = [b]
            else:
                edge_idx[a].append(b)
            if not b in edge_idx:
                edge_idx[b] = [a]
            else:
                edge_idx[b].append(a)
            # edge_idx[a][b] = 1
            # edge_idx[b][a] = 1
        f.close()
    return edge_idx, set(edge_unique), nodes, deg

def heavy_hitters(nodes, deg, edges, num_edges):
    """
    Return all the node that considered as heavy hitters
    """
    lst = []
    for node in nodes:
        degree = deg[node]
        if degree >= sqrt(num_edges):
            lst.append(node)
    return lst

def count_hh_triangles(heavy_hitters_lst, edge_idx):
    cnt = 0
    for a,b,c in itertools.combinations(heavy_hitters_lst, 3):
        if a in edge_idx[b] and c in edge_idx[b] and c in edge_idx[a]:
            cnt += 1
    return cnt

def isSmaller(node1, node2, deg):
    if deg[node1] < deg[node2]:
        return True
    if deg[node1] == deg[node2] and node1 < node2:
        return True
    return False

def count_other_triangles(heavy_hitters_lst, deg, edge_idx, edges):
    cnt = 0
    for edge in edges:
        a,b = edge
        if a in heavy_hitters_lst and b in heavy_hitters_lst:
            continue
        if isSmaller(a,b, deg):
            v1, v2 = a, b
        else:
            v1, v2 = b, a
        v1_adj = edge_idx[v1]
        for u in v1_adj:
            if u == v2:
                continue
            if u in edge_idx[v2] and isSmaller(v1, u, deg) and isSmaller(v2, u, deg):
                cnt += 1
    return cnt

start = time.time()
edge_idx, edges, nodes, deg = loadData(sys.argv[1])
m = len(edges) # number of edges

hh_lst = heavy_hitters(nodes, deg, edges, m)
total = count_hh_triangles(hh_lst, edge_idx) + count_other_triangles(hh_lst, deg, edge_idx, edges)
print(total)

end = time.time()
# print(f"Total Time Execute = {end - start}s")



