#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
import re
import time
# maps words to their counts
movieID = []
Rating = []
result_rating = []
word_counter = 0
counter = 0
presentID = 0
user_counter = 0
userID = []
Average_rating = [] 
result_average_rating = []
presentUserID = 0
tme = 0
## Rating Genere 
presentGenere = ""
rating_genere = []
average_rating_genere = []
Genere_list = []
# input comes from STDIN
for line in sys.stdin:
    
    try :    
        line = line.strip()
        
        genere, array = line.split(';')

        array = re.split('[|,|]',array)
        rating = float((array[0].strip().split('['))[1])
        tme = float((array[1].strip().split(']'))[0])

      #print (genere , rating, tme )

        #print (rating , genere)
        if (counter == 0) :
            presentGenere = genere
            Genere_list.append(presentGenere)
            #movieID.append(presentID)
            rating_genere.append(rating)
            counter += 1
        else :
        		#print(int(delay))
            if (presentGenere != genere) :
                
                
                presentGenere = genere
                Genere_list.append(presentGenere)
                average_rating_genere.append(np.mean(rating_genere))
                rating_genere = []
            rating_genere.append(rating)
        #counter += 1
    except ValueError:
              pass
#print(len(average_rating_genere) ,' ', len(Genere_list
              
index = average_rating_genere.index(max(average_rating_genere))
print ('index', index)
print ('lenght', len(Genere_list) )
print ("Best Ratio Genere ", Genere_list[index])
print( 'rating', max(average_rating_genere) ) 

end = time.time() - tme
print ("total program time :", end )