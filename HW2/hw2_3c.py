import numpy as np
import re
import sys
import math
import csv
import os


# Load Data into dictionary
def load_data(file):
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
	return dct

data = load_data(sys.argv[1])

# creating lists for unique user id's and movie id
unique_users = list(set(data["user_id"]))
unique_movies = list(set(data["movie_id"]))

# user list and movie list
user_lst = data["user_id"]
movie_lst = data["movie_id"]
rating_lst = data["rating"]

# Creating a dictionary to map movie id to their corresponding index.
movies_dict = {unique_movies[i] : i for i in range(len(unique_movies))}

# Create a utility matrix (movies x users)
utility_matrix = np.asarray([[np.nan for j in range(len(unique_users))] for i in range(len(unique_movies))])
for i in range(len(rating_lst)):
	utility_matrix[movies_dict[movie_lst[i]]][user_lst[i]-2] = rating_lst[i]


mask = np.isnan(utility_matrix) # mask every "nan" value
masked_arr = np.ma.masked_array(utility_matrix, mask)
temp_mask = masked_arr # (movies x users)
zeros = np.zeros(temp_mask.shape[1]) 
filled_matrix = temp_mask.filled(zeros) # filled all "nan" value by zeros
filled_matrix = filled_matrix.T # (users x movies)


def matrix_factorization(M, P, Q, K, num_epochs=1, l_rate=0.0002, beta=0.02):
    Q = Q.T
    for step in range(num_epochs):
        for i in range(len(M)):
            for j in range(len(M[i])):
                if M[i][j] > 0:
                    eij = M[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + l_rate * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + l_rate * (2 * eij * P[i][k] - beta * Q[k][j])
        l = 0
        for i in range(len(M)):
            for j in range(len(M[i])):
                if M[i][j] > 0:
                    l = l + pow(M[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        l = l + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
    return P, Q.T


N = len(filled_matrix)
M = len(filled_matrix[0])
K = 3 # number of features

U = np.random.rand(N,K)
V = np.random.rand(M,K)

nU, nV = matrix_factorization(filled_matrix, U, V, K)
nM = np.dot(nU, nV.T)

dct_test = load_data(sys.argv[2])
L = len(dct_test["user_id"])
preds = []
for i in range(L):
    user_id = dct_test["user_id"][i]
    movie_id = dct_test["movie_id"][i]
    if movie_id in unique_movies:
	    pred_rating = nM[user_id-2][movies_dict[movie_id]]
	    preds.append(pred_rating)
    else:
    	preds.append(0)
# print(preds)
  
out_file = "output.txt"
answ = os.path.exists(out_file)
with open(out_file, "a" if answ else "w") as f:
    for idx, r in enumerate(preds):
        f.write(str(dct_test["user_id"][idx]) + "," + str(dct_test["movie_id"][idx]) + "," + str(r) + "," + str(dct_test["time_stamp"][idx]))
        f.write("\n")

