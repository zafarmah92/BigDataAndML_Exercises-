#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
import re
# maps words to their counts
movieID = []
Rating = []
result_rating = []
word_counter = 0
counter = 0
presentID = 0
user_counter = 1
userID = []
Average_rating = [] 
result_average_rating = []
presentUserID = 0


finalAbove40_users = [] 
finalAbove40_average_rating = []
# input comes from STDIN

for line in sys.stdin:
    try :    
        line = line.strip()
        _id , array = line.split(';') 
        _id = int(_id)
        array = re.split('[|,|]',array)
        rating = float((array[0].strip().split('['))[1])
        tme = float((array[1].strip().split(']'))[0])
        #print(title)
    
        if (counter == 0) :
            presentUserID = _id 
            #presentTitle = title
            #movieTitle.append(presentTitle)
            Rating.append(rating)
            counter += 1
            user_counter = 1
            
        else :
        	
            if (presentUserID != _id) :
                if (user_counter > 40) :
                    finalAbove40_users.append(presentUserID)
                    finalAbove40_average_rating.append(np.mean(Rating))
                    print ("flick")
                print( 'change ', _id , '  rating ' , rating )    
                presentUserID = _id
                user_counter = 1
                Rating = []
            else :
                Rating.append(rating)
                user_counter += 1


 
    except ValueError:
              #print('\\\\\\\\\\\\\\ERROR' ) 
              #print(arrival)
              pass


if (user_counter > 40) :
    finalAbove40_users.append(presentUserID)
    finalAbove40_average_rating.append(np.mean(Rating))
              
print(" Total Number of  Users who rated more than 40 times ",len(finalAbove40_users) )
#print(len(finalAbove40_average_rating))
x = np.argmin(finalAbove40_average_rating)

print (" User with lowest ID : ", finalAbove40_users[x], " with rating " , finalAbove40_average_rating[x])
