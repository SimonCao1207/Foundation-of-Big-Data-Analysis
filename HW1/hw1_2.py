import sys
# import time
def create_baskets(fileName):
	"""
	Return baskets of items
	"""
	baskets = []
	with open(fileName, "r") as file:
		for line in file:
			baskets.append(line.split())
	return baskets

def freq_items(baskets, threshold=200):
	"""
	Return distinct items set and frequent items 
	"""
	items = {}
	freqItems = [] 
	for basket in baskets:
		for item in basket:
			if not item in items:
				items[item] = 1
			else:
				items[item] += 1
				if items[item] >= threshold and item not in freqItems:
					freqItems.append(item)
	return items, freqItems

def map_items(items, freqItems):
	"""
	Map item to index number and 
	Map index number to item for retrieving purpose.
	"""
	item_to_idx = {}
	idx_to_item = {}
	idx = 1
	for item in items:
		if item not in freqItems:
			item_to_idx[item] = 0
		else:
			item_to_idx[item] = idx
			idx_to_item[idx] = item
			idx += 1
	return item_to_idx, idx_to_item


def freq_pairs(baskets, item_to_idx, idx_to_item,  mfreqItems, threshold=200):
	"""
	Return number of frequent pairs and list of frequent pairs and their frequency.
	"""
	# Using triangular matrix to store counts of pairs.
	triangular_matrix = [[0 for i in range(mfreqItems+1)] for j in range(mfreqItems+1)] 
 	
	for basket in baskets:
		freqItems = []  # storing frequent items in each basket
		for item in basket:
			if not item_to_idx[item] == 0: # if index of item is 0 -> meaning is not frequent
				freqItems.append(item)
		combinations = [] # storing combination of pairs in frequent items set.
		for i in range(len(freqItems)-1):
			for j in range(i+1, len(freqItems)):
				combinations.append((freqItems[i],freqItems[j]))
		for p1, p2 in combinations:
			i1, i2 = item_to_idx[p1], item_to_idx[p2]	
			if i1 < i2:
				triangular_matrix[i1][i2] += 1
			else: 
				triangular_matrix[i2][i1] += 1
	cnt = 0
	freqPairs = [] # storing frequent pairs and their frequency stored in triangular matrix
	for i in range(mfreqItems):
		for j in range(i+1, mfreqItems+1):
			if triangular_matrix[i][j] >= threshold:
				cnt += 1
				freqPairs.append((idx_to_item[i], idx_to_item[j], triangular_matrix[i][j]))
	return cnt, freqPairs


def sort_freq_pairs(freqPairs):
	"""
	Return sorted frequent pairs based on their count (in descending order)
	"""
	freqPairs = sorted(freqPairs, key=lambda pair: -pair[2])
	return freqPairs

# start = time.time()	
baskets = create_baskets(sys.argv[1])

items, freqItems = freq_items(baskets)
item_to_idx, idx_to_item = map_items(items, freqItems)
cnt, freqPairs = freq_pairs(baskets, item_to_idx, idx_to_item, len(freqItems))
sorted_freqPairs = sort_freq_pairs(freqPairs)

print(len(freqItems)) # number of frequent items
print(cnt) # number of frequent pairs
# Top-10 frequent pairs
for pair in sorted_freqPairs[:10]:
	print(f"{pair[0]}\t{pair[1]}\t{pair[2]}")
# end = time.time()
# print("Time", end-start)