import re
import numpy as np
import sys
import time
def split_documents(fileName):
	"""
	split the content in txt file into documents
	"""
	with open(fileName, 'r') as file:
		content = file.read()
		documents = content.split('\n')
	return documents

def _isPrime(n):
	"""
	return if n is a prime number
	"""
	if n == 2 or n == 3: return True
	if n < 2 or n%2 == 0: return False
	if n < 9: return True
	if n%3 == 0: return False
	r = int(n**0.5)
	f = 5
	while f <= r:
		if n % f == 0: return False
		if n % (f+2) == 0: return False
		f += 6
	return True 

def randomHashCoeffs(n, numHashes):
	"""
	return list of coefficients of hasfunction (ax+b)%c 
	c: smallest prime number larger than or equal to n
	a,b : random between [0, c-1]
	"""

	firstPrime = n # first prime number larger >= n
	while not _isPrime(firstPrime):
		firstPrime += 1
	randlst = []
	for i in range(numHashes):
		idx = np.random.randint(0, firstPrime-1)
		while idx in randlst:
			idx = np.random.randint(0, firstPrime-1)
		randlst.append(idx)
	return randlst, firstPrime

def shingling_articles(documents, k_shingles):
	articleNames = [] # store all articleNames
	curID = 0 	#current row
	allShingle = {} # map shingle -> rowID
	shinglesMap = {} # map article name --> list of shingles ID

	for document in documents:
		article_id, rest = document.split(" ", 1)
		articleNames.append(article_id)
		rest = re.sub(r'[^A-Za-z0-9 ]+', '', rest).lower() #ignore non-alphabetic character

		shingleIDs = set() # set of shingle IDs in this document

		for i in range(len(rest)-k_shingles+1):
			shingle = rest[i:i+k_shingles]
			if not shingle in allShingle:
				allShingle[shingle] = curID
				curID += 1
			shingleIDs.add(allShingle[shingle])
		
		shinglesMap[article_id] = shingleIDs

	totalShingle = len(allShingle)

	return shinglesMap, totalShingle, articleNames

def generate_signatures(articles, map, nHashes, nDocs):
	signatures = [] # signature vectors for the whole data
	a_randlst, firstPrime = randomHashCoeffs(nDocs, nHashes)
	b_randlst, firstPrime = randomHashCoeffs(nDocs, nHashes)

	for name in articleNames:
		shingleIDs = m[name]
		signature = []
		for i in range(nHashes):
			minHash = float("inf")
			for shingleID in shingleIDs:
				hashCode = (a_randlst[i]*shingleID + b_randlst[i]) % firstPrime
				if hashCode < minHash:
					minHash = hashCode
			signature.append(minHash)
		signatures.append(signature)
	return signatures

def calculate_JSim(X, Y):
	"""
	Calculate Jacred similarity between 2 sets:
	"""
	nHashes = len(X)
	cnt = 0
	for i in range(nHashes):
		if X[i] == Y[i]:
			cnt +=1
	return cnt/nHashes

def _generate_pairs(bucket, newID, candidatePairs):
	"""
	This is used for find_candidate_pairs function
	To : update candidate pairs set when add new ID into the bucket
	"""
	for idx in bucket:
		newPair = (idx, newID)
		candidatePairs.add(tuple(sorted(newPair)))
	return candidatePairs

def find_candidate_pairs(signatures, r=20, b=6):
	if not len(signatures[0]) == b*r:
		print("The number of bands does not devide the signature vector evenly.")
		sys.exit(1)

	candidatePairs = set()

	for band in range(0, b):
		buckets = {}
		for idx, s in enumerate(signatures):
			band_sinatures = []
			for row in range(band*r, (band+1)*r):
				band_sinatures.append(s[row])

			hashValue = tuple(band_sinatures) # dictionary can hash tuple but not list.

			if hashValue not in buckets:
				buckets[hashValue] = [idx]
			else:
				candidatePairs = _generate_pairs(buckets[hashValue], idx, candidatePairs)
				buckets[hashValue].append(idx)
	
	return candidatePairs


def suggested_similar_pairs(articleNames,candidatePairs, signatures, threshold=0.9):
	"""
	From candidate Pairs, calculate the actual similarity between signatures vectors
	Return pairs that have Jacred similarity >= threshold
	"""
	similarPair = []
	for a,b in candidatePairs:
		jsim = calculate_JSim(signatures[a], signatures[b])
		if jsim >= threshold:
			similarPair.append((articleNames[a],articleNames[b]))
	return similarPair

start = time.time()
fileName = sys.argv[1]
documents = split_documents(fileName)
m, totalShingle, articleNames = shingling_articles(documents, k_shingles=3)
numDocs = len(m)
signatures = generate_signatures(articles=articleNames, map=m, nHashes=120, nDocs=numDocs)

candidatePairs = find_candidate_pairs(signatures, r=20, b=6)
similarPairs = suggested_similar_pairs(articleNames, candidatePairs, signatures, threshold=0.9)

for pair in similarPairs:
	print(f"{pair[0]}\t{pair[1]}")
end = time.time()
print(end-start)