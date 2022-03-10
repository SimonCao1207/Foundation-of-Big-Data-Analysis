import sys
import itertools
thres_hold = 200

cnt_item = {}
freq_items = {}

with open(sys.argv[1], 'r') as lines:
	for line in lines:
		for i in set(line.strip().split(" ")):
			if i not in cnt_item:
				cnt_item[i] = 0
			cnt_item[i] += 1
			if cnt_item[i] >= thres_hold:
				freq_items[i] = True

baskets = []
with open(sys.argv[1], "r") as file:
	for line in file:
		baskets.append(line.strip().split(" "))

items = {}
freqItems = [] 
for basket in baskets:
	for item in basket:
		if not item in items:
			items[item] = 1
		else:
			items[item] += 1
			if items[item] >= thres_hold and item not in freqItems:
				freqItems.append(item)


print(len(freq_items))
print(len(freqItems))