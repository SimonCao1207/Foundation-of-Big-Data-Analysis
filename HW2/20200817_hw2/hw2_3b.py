import numpy as np
import re
import sys
import math
import csv
# import time
import os

# Load Data into dictionary
def load_data(file):
	dct = {"user_id" : [] , "movie_id" : [] , "rating": [], "time_stamp" : []}
	with open(file, "r") as f:
		for l in f:
			user_id, movie_id, rating, timestamp = l.split(",")
			user_id, movie_id, timestamp = int(user_id), int(movie_id), int(timestamp)
			rating = float(rating)
			dct["user_id"].append(user_id)
			dct["movie_id"].append(movie_id)
			dct["rating"].append(rating)
			dct["time_stamp"].append(timestamp)
		f.close()
	return dct

def norm(vec):
	return np.sqrt(np.sum(vec * vec))

# Item-based	
def top_cosine_item(data, index, top_n=10):
	movie_col = data[:, index]
	similarity =  []
	for i in range(len(unique_movies)):
		if i == index:
			continue
		v = data[:, i]
		if norm(movie_col) * norm(v) == 0:
			continue
		cosine_dist = np.dot(movie_col, v) / (norm(movie_col) * norm(v))
		similarity.append((-cosine_dist, i))
	sort_indexes = [i for d, i in sorted(similarity)]
	return sort_indexes[:top_n]
  

def item_based_rec(user_id):
	# user_id = 600
	index = user_id-2
	user_row = utility_matrix[index, :]
	preds = []
	for idx, rating in enumerate(user_row):
		if np.isnan(rating) and unique_movies[idx] in range(1, 1001):
			colIdx = top_cosine_item(normal_matrix, idx)
			rates = [utility_matrix[index][i] for i in colIdx if not np.isnan(utility_matrix[index][i]) and not utility_matrix[index][i] == 0]
			if len(rates) == 0:
				continue
			else:	
				pred_rating = sum(rates)/len(rates)
			# print((unique_movies[idx], pred_rating))
			# utility_matrix[index][idx] = pred_rating
			preds.append((pred_rating, -unique_movies[idx]))
	preds = sorted(preds, reverse=True)
	top_5 = [(-i, r) for r, i in preds[:5]]	
	return top_5

# User based
def top_cosine_user(data, index, top_n=10):
	user_row = data[index, :]
	similarity =  []
	for i in range(len(unique_users)):
		if i == index:
			continue
		v = data[i, :]
		if norm(user_row) * norm(v) == 0:
			continue
		cosine_dist = np.dot(user_row, v) / (norm(user_row) * norm(v))
		similarity.append((-cosine_dist, i))
	sort_indexes = [i for d, i in sorted(similarity)]
	return sort_indexes[:top_n]

def user_based_rec(user_id=600):
	# user_id = 600
	index = user_id-2
	user_row = utility_matrix[index, :]
	sim_users = top_cosine_user(normal_matrix, index)
	preds = []
	for idx, rating in enumerate(user_row):
		if np.isnan(rating) and unique_movies[idx] in range(1, 1001):
			rates = [utility_matrix[i][idx] for i in sim_users if not np.isnan(utility_matrix[i][idx])]
			if len(rates) == 0:
				pred_rating = 0
			else:
				pred_rating = sum(rates)/len(rates)
			# utility_matrix[index][idx] = pred_rating
			preds.append((pred_rating, -unique_movies[idx]))
	preds = sorted(preds, reverse=True)
	top_5 = [(-i, r) for r, i in preds[:5]]	
	return top_5

def print_result(user_id=600, method="user-based"):
	if method == "user-based":
		rec_movie_ids = user_based_rec(user_id)
		for u, v  in rec_movie_ids:
			print(f"{u}\t{v}")
	elif method == "item-based":
		rec_movie_ids = item_based_rec(user_id)
		for u, v  in rec_movie_ids:
			print(f"{u}\t{v}")


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


# Normalization, subtract from each rating the average rating of that user
mask = np.isnan(utility_matrix)
masked_arr = np.ma.masked_array(utility_matrix, mask)
tmp_mask = masked_arr # (movies x users)
rating_means = np.mean(tmp_mask, axis=0) # average rating of each user
fillMa = tmp_mask.filled(rating_means)
fillMa = fillMa.T # (users x movies)
normal_matrix = fillMa - rating_means.data[:,np.newaxis] # (users x movies)
utility_matrix = utility_matrix.T # (users x movies)

#USER-BASED result:
print_result(600, method="user-based")

#ITEM-BASED result:
print_result(600, method="item-based")