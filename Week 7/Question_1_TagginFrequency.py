from pyspark import SparkContext
import numpy as np


user_time_stamp = []
avg_user_stamp = []
user = [] 
 
sc = SparkContext("local","Simple App")
 
logData = sc.textFile("tags.dat")
 
extracted_userData = logData.map(lambda x:x.split('::')).map(lambda y:(int(y[0]),int(y[3]))).collect()
 
print(len(extracted_userData))

mat = np.matrix(extracted_userData)

v = len(np.array(mat[:,0]))
print("length of v ", v)
unique_users = np.zeros(v)
time_stamps = np.zeros(v)
array_counter = 0
all_time_stamps = []
presentID = 0 
counter = 0

for i in range(0,v):
    _id = mat[i,0]
    #print(_id)

    if(counter == 0):
        counter += 1
        print("this is counter")
        presentID = _id
        array_counter += 1
        user.append(presentID)
        user_time_stamp.append(mat[i,1])
           
    else :
        
        if (presentID != _id ):
            user.append(presentID)
            all_time_stamps.append(list(np.sort(user_time_stamp)))
            user_time_stamp = []
            array_counter += 1
            presentID = _id
            
        user_time_stamp.append(mat[i,1])
                


print(array_counter)


#user.append(presentID)
all_time_stamps.append(list(np.sort(user_time_stamp)))
user_time_stamp = []

print("shape of stamps ",np.shape(all_time_stamps))
print("few data",all_time_stamps[0:10])
print(type(all_time_stamps))

## Now next questions 

user_tagging_frequecies = []
user_tagging_time_stamps = []

taggin_session = 1800

temp_stamp = 0,
temp_frequency_session = 0

_fileWrite = open("taging.txt","w")

for i in range(0,len(user)):
    print("lenght:", len(all_time_stamps[i]))
  
    for j in range(0,len(all_time_stamps[i])):
       
        if (len(all_time_stamps[i]) == 1):
            user_tagging_frequecies.append(1) 
        else :

            if (j == 0):
                temp_stamp = all_time_stamps[i][j]
                temp_frequency_session = 1
            else :
                if(all_time_stamps[i][j] >= (temp_stamp + 1800)):

                    user_tagging_frequecies.append(temp_frequency_session) 
                    temp_stamp = all_time_stamps[i][j]
                    temp_frequency_session = 1
                else :   
                   temp_frequency_session += 1
                
        
    if (temp_frequency_session != 0):
        user_tagging_frequecies.append(temp_frequency_session) 
                    
    #user_tagging_frequecies.append(temp_frequency_session) 
    
    print("user ID ",user[i], " tagging_sessions:", len(user_tagging_frequecies)," std:",np.std(user_tagging_frequecies)," mean:",np.mean(user_tagging_frequecies))
    print(user_tagging_frequecies)
    if (np.std(user_tagging_frequecies)>2):
        print("user ID ",user[i], " tagging_sessions:", len(user_tagging_frequecies)," std:",np.std(user_tagging_frequecies)," mean:",np.mean(user_tagging_frequecies), "\n")
        _fileWrite.write("UserID %s, TagginSession %s, STD %s, mean %s,\n" % (user[i], len(user_tagging_frequecies), np.std(user_tagging_frequecies), np.mean(user_tagging_frequecies)))   
    user_tagging_frequecies = []
    temp_frequency_session = 0
    if (i == 1000):
        break
                

