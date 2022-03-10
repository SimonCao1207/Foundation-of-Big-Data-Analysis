from pyspark import SparkConf, SparkContext
import sys
# import time

def user_and_friends(line):
	"""
	Return user and list of his friends 
	"""
	split = line.split()
	user_id = int(split[0])
	friends = list(map(lambda x: int(x), split[1].split(","))) if len(split)>1 else []
	return user_id, friends

def create_map(u_f):
	"""
	Map key (user, his friend) -> 0. 
	Map key (friend_i, friend_j) -> 1 where friend_i and friend_j are friends of user.
	"""
	user, friends = u_f
	connect_map = []
	combinations = []
	for i in range(len(friends)-1):
		for j in range(i+1, len(friends)):
			combinations.append((friends[i],friends[j]))
	for friend in friends:
		key = (user, friend) if user < friend else (friend, user)
		connect_map.append((key, 0))
	for friend_1, friend_2 in combinations:
		key = (friend_1, friend_2) if friend_1 < friend_2 else (friend_2, friend_1)
		connect_map.append((key, 1))
	return connect_map

def sort_recomendation(recs):
	"""
	Sorted by their counts in descending order and if tie, sort by the first user ID integer in ascending order 
	and then the second.

	Rerturn top-10
	"""
	recs.sort(key=lambda x: (-x[1], x[0]))
	return recs[:10]

# start = time.time()
conf = SparkConf()
sc = SparkContext(conf=conf)
lines = sc.textFile(sys.argv[1])
user_friends = lines.map(user_and_friends) # map each line in lines -> user, list of friends
mutual_friends = user_friends.flatMap(create_map) # (user,friend):1 and (friend, friend):0
filter_mutual_friends = mutual_friends.groupByKey().filter(lambda x: 0 not in x[1]) # filter all those have 0's in the list of values
cnt = filter_mutual_friends.map(lambda x: (x[0], sum(x[1]))) # sum all the values of mutual friends
mutual_recommends = cnt.collect()
ans = sort_recomendation(mutual_recommends) #sort and pick top-10
for item in ans:
	users, cnt = item
	user1, user2 = users
	print(f"{user1}\t{user2}\t{cnt}")
# end = time.time()
# print("time", end-start)


