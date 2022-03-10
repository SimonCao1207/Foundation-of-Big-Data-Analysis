import re
import sys
from pyspark import SparkConf, SparkContext

conf = SparkConf()
sc = SparkContext(conf=conf)
lines = sc.textFile(sys.argv[1])
words = lines.flatMap(lambda l: re.split(r'[^\w]+', l))
pairs = words.map(lambda w: (w[0].lower(), 1) if len(w) else (w, 0))
counts = pairs.reduceByKey(lambda n1, n2: n1 + n2)
lst  = counts.collect()
lst = sorted(lst)
dt = {k:v for k,v in lst if k.isalpha()}
alphabet_string = 'abcdefghijklmnopqrstuvwxyz'
for l in list(alphabet_string):
	if l not in dt:
		freq = 0
	else:
		freq = dt[l]
	print(f"{l}\t{freq}")
sc.stop()
