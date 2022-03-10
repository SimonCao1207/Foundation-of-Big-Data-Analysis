import numpy as np
import re
import sys
from math import *
import csv
import time
import os

# Load Data into dictionary
def split_data(file, prop_train=0.95):
    dct = {"user_id" : [] , "movie_id" : [] , "rating": [], "time_stamp" : []}
    with open(file, "r") as f:
        for l in f:
            user_id, movie_id, rating, timestamp = l.split(",")
            user_id, movie_id, timestamp = int(user_id), int(movie_id), int(timestamp)
            rating = float(rating) if not rating == ''  else ''
            dct["user_id"].append(user_id)
            dct["movie_id"].append(movie_id)
            dct["rating"].append(rating)
            dct["time_stamp"].append(timestamp)
        f.close()
    len_data = len(dct["user_id"])
    num_train = int(prop_train * len_data)
    dct_train = {"user_id" : [dct["user_id"][i] for i in range(num_train)] , 
                "movie_id" : [dct["movie_id"][i] for i in range(num_train)] , 
                "rating": [dct["rating"][i] for i in range(num_train)], 
                "time_stamp" : [dct["time_stamp"][i] for i in range(num_train)]}

    dct_valid = {"user_id" : [dct["user_id"][i] for i in range(num_train, len_data)] , 
                "movie_id" : [dct["movie_id"][i] for i in range(num_train, len_data)] , 
                "rating": [dct["rating"][i] for i in range(num_train, len_data)], 
                "time_stamp" : [dct["time_stamp"][i] for i in range(num_train, len_data)]}

    return dct_train, dct_valid

def rmse(ground_true, preds):
    l = len(ground_true)
    sqr = [(ground_true[i]-preds[i])**2 for i in range(l)]
    return sqrt(sum(sqr)/l)

def preprocessing(matrix):
    mean_user = np.mean(matrix, axis=1)
    mean_item = np.mean(matrix, axis=0)
    nM = matrix - mean_user[:, np.newaxis]
    nM = nM - mean_item[np.newaxis, :]
    return nM, mean_user, mean_item

def inverseProcess(matrix, mean_user, mean_item):
    nM = matrix
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 0:
                continue
            else:
                s = mean_user[i] + mean_item[j]
                nM[i][j] += s
    return nM



# Devide 5% of data for validation
train_data, valid_data = split_data("ratings.txt")


# creating lists for unique user id's and movie id
unique_users = list(set(train_data["user_id"]))
unique_movies = list(set(train_data["movie_id"]))

# user list and movie list
user_lst = train_data["user_id"]
movie_lst = train_data["movie_id"]
rating_lst = train_data["rating"]

# Creating a dictionary to map movie id to their corresponding index.
movies_dict = {unique_movies[i] : i for i in range(len(unique_movies))}
user_dict = {unique_users[i] : i for i in range(len(unique_users)) }

# Create a utility matrix (movies x users)
utility_matrix = np.asarray([[np.nan for j in range(len(unique_users))] for i in range(len(unique_movies))])
for i in range(len(rating_lst)):
    utility_matrix[movies_dict[movie_lst[i]]][user_dict[user_lst[i]]] = rating_lst[i]

# print(utility_matrix.T)  

mask = np.isnan(utility_matrix)
temp_mask = np.ma.masked_array(utility_matrix, mask) # (movies x users)
zeros = np.zeros(temp_mask.shape[0]) 
new_mask, mean_user, mean_item = preprocessing(temp_mask.T)


filled_matrix = new_mask.filled(zeros) # (users x movies)

s = 0
cnt = 0
for i in range(len(filled_matrix)):
  for j in range(len(filled_matrix[0])):
      if filled_matrix[i][j] == 0:
          continue
      else:
          s += filled_matrix[i][j]
          cnt += 1

avg_nonblank = s/cnt

def matrix_factorization(R, P, Q, K, steps=10, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        loss = e/len(train_data["user_id"])
        print(f"epoch {step}: {loss}")
        if e < 0.001:
            break
    return P, Q.T


R = filled_matrix
N = len(R)
M = len(R[0])
K = 3

P = np.full((N,K), avg_nonblank)
Q = np.full((M,K), avg_nonblank)

nP, nQ = matrix_factorization(R, P, Q, K)
nR = np.dot(nP, nQ.T)

pred_matrix = inverseProcess(nR, mean_user, mean_item)


