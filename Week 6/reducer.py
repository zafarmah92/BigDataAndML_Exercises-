
#!/usr/bin/python

import sys
import time 
import numpy as np
import re, math

current_word = None
current_count = 1
counter = 0
time_end = 0

word_count_document = []
present_document = 0
distintive_words = []
total_words_docs = np.zeros(3)
_non_redundant = []

def _remove_redundencies():

    for i in range(0,len(word_count_document)):
        wd = word_count_document[i][0]
        _cnt = word_count_document[i][1]
        dc = word_count_document[i][2]
        _new_count = 1
        check = 0
        if (len(_non_redundant) > 0):
           for k in range(0,len(_non_redundant)):
                if (_non_redundant[k][0] == wd and dc == _non_redundant[k][2]) :
                    check = 1
        if (check == 0 or len(_non_redundant) == 0) :
            for j in range(0,len(word_count_document)):
                if (word_count_document[j][0] == wd and dc == word_count_document[j][2]):
                    _new_count += 1 

            _non_redundant.append([wd,_new_count,dc])

    print("non redundant \n\n\n\n",_non_redundant)



def _word_in_all_docs(_word):
    _word_doc_count = np.zeros(3)
    
    #print(_array[1][2])

    for j in range(0,len(_non_redundant)):
         

        if (_word == _non_redundant[j][0]):
            
            _word_doc_count[_non_redundant[j][2]] = 1

    return sum(_word_doc_count)


def _tf (wrd,cnt,ttl_doc_count):
    #print(wrd," ",cnt," ",ttl_doc_count,"   TF :: ", cnt/ttl_doc_count)
    return cnt/ttl_doc_count

def _idf(ttl_docs , word_number_of_docs):
    var = math.log(ttl_docs/word_number_of_docs)
    return var





for line in sys.stdin:

    #word, count , tme = line.strip().split('\t')
    word, array = line.strip().split('\t')
    #count, tme , doc = array.split(',')
    array = re.split('[|,|]',array)
    doc = int ((array[2].strip().split(']'))[0])
    tme = float(array[1].strip())
    total_words_docs[doc] += 1
    
    #print(array,'  ',doc)
    #print(count,':', tme,':',doc)
    #doc = int(doc)

    try :
        if (counter == 0):
            current_word = word
            present_document = doc
            distintive_words.append(current_word)
            #print (count , tme , doc)
            #print (present_document[0] , present_document[1] , present_document[2], present_document[3])
            time_end = float(tme)
            print ('time end ',time_end,' ',type(time_end))   
            counter += 1
        else :
                
            if (current_word == word and present_document == doc ):
                current_count += 1
            elif (present_document != doc and current_word == word) :
                
                word_count_document.append([current_word,current_count,present_document])
                print ("%s\t%s\t%s" % (current_word, current_count,present_document))
                present_document = doc
                current_count = 1
                
            else :    
                word_count_document.append([current_word,current_count,present_document])

                print ("%s\t%s\t%s" % (current_word, current_count,present_document))
                current_word = word
                distintive_words.append(current_word)
                current_count = 1     

    except ErrorValue:
         pass



word_count_document.append([current_word,current_count,present_document])
print ("%s\t%s\t%s" % (current_word, current_count,present_document))


_remove_redundencies() 

for i in range(0,len(_non_redundant)):
       
   tf = _tf(_non_redundant[i][0],_non_redundant[i][1], total_words_docs[_non_redundant[i][2]])
   #print (tf)
   idf = _idf(len(total_words_docs), _word_in_all_docs(_non_redundant[i][0]))

   print (" TFIDF of word :", _non_redundant[i][0], "  :  ",tf*idf, " Document :: ", _non_redundant[i][2])

end = time.time() - time_end
print ("total program time :", end )
