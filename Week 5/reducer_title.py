#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
import time
import re

# maps words to their counts
movieID = []
result_rating = []
word_counter = 0
counter = 0
presentID = 0
user_counter = 0
userID = []
result_average_rating = []
presentUserID = 0


presentTitle = None
Average_rating = [] 
Rating = []
movieTitle = []
# input comes from STDIN
for line in sys.stdin:
    try :    
        line = line.strip()
        title , array = line.split(';') 
        array = re.split('[|,|]',array)
        rating = float((array[0].strip().split('['))[1])
        tme = float((array[1].strip().split(']'))[0])
        #print(title)
    
        if (counter == 0) :
            presentTitle = title
            movieTitle.append(presentTitle)
            Rating.append(rating)
            counter += 1

        else :

            if (presentTitle != title) :

                presentTitle = title
                Average_rating.append(np.mean(Rating))
                Rating = []
                movieTitle.append(presentTitle)


            Rating.append(rating)
        
 
          
    except ValueError:
            pass

Average_rating.append(np.mean(Rating))
#print (Average_rating)
#print (movieTitle)
index = np.argmax(Average_rating)
print ( " Highest Rated Movie : ", movieTitle[index], " , with Rating : " , Average_rating[index] )
 
end = time.time() - tme
print ("total program time :", end )
#print (word2count)
#resultList.append([np.min(delayList) , np.max(delayList),np.mean(delayList)]) 

#print(average_arrival)
#print(len(resultList))
#print(len(wordList))
#print(len(average_arrival))
#avgArrival, wordList = (list(t) for t in zip(*sorted(zip(average_arrival, wordList))))
# print('length of rating ', len(result_average_rating))
# print('length of Users ', len(userID))
# print (len(result_rating))
# print (len(movieID))
# print('max average rating ', np.max(result_rating))
# print ( result_rating.index(max(result_rating)))
# print ("Movie ID ", movieID[result_rating.index(max(result_rating))])

# try : 
#     value = sorted(set(result_average_rating))[1]
#     getIndex = result_average_rating.index(value)
#     print("user id who is second minium ",userID[getIndex])
# except : 
#     print ("there is no user who rated 40 times")
#     pass
# #print(wordList[:10])