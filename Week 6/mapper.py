#!/usr/bin/env python


import sys
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import time 

start_time = time.time()
file = 0


for line in sys.stdin:

	line = line.strip()

	if (line == "next_file"):
		#print("its next file",line)
		file += 1
		

	sentence = line.lower()
	
	words = word_tokenize(sentence)

	words =[word.lower() for word in words if word.isalpha()]

	_stopwords = set(stopwords.words('english'))

	filtered_sentence = []
	
	if (len(words) != 0):
		for w in words:
			if w not in _stopwords :
				filtered_sentence.append(w)
				print ('%s\t%s' % (w,[1,start_time,file]))


