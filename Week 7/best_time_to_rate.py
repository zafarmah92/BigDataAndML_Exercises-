from pyspark import SparkContext
import numpy as np
import time 

sc = SparkContext("local","Simple App")
 
logData = sc.textFile("/home/zfar/ml-10M100K/ratings.dat")

extracted_userData = logData.map(lambda x:x.split('::')).map(lambda y:(y[0],int(y[3]))).collect()

print(len(extracted_userData))
print(extracted_userData[1:5])

mat = np.matrix(extracted_userData)

Users = []
user_rating_time = []
presentUser = 0
user_counter = 0
_parsed_rating_time = []

_searchedUserID = input("Enter ID : ")
print(_searchedUserID)


_file_write = open("best_time_to_rating.txt","w")


for i in range(0,len(extracted_userData)):

	_usr = mat[i,0]
	_rating_time = int(mat[i,1])

	if (i == 0):
	
		presentUser = _usr
		user_counter = 1
		Users.append(presentUser)
		#user_rating_time.append(_rating_time)
	
	else :
		if (presentUser != _usr):

			print(presentUser," ",np.mean(user_rating_time))
			#_file_write.write("userID: %s  Average Rating :%s\n" %(presentUser , np.mean(user_rating_time)))
			_hours = np.mean(user_rating_time)
			#_hours = (_hours / 60)/60  ## divding by second to get minutes  &   minutes to get hours , 
			_time = time.strftime("%H:%M:%S", time.gmtime(_hours))
			_time = _time.split(":")

			_parsed_rating_time.append(_time)

			if (int(_time[0]) < 6 ): 
				print("User ", presentUser, " usually rates Early morning before 6")
				_file_write.write(" user %s usually rates Early morning before 6,   time ::%s \n" % (presentUser,_time))
			elif (int(_time[0]) < 12 ):
				print("User ", presentUser, " usually rates before noon before 12 ")

				_file_write.write(" user %s usually rates before noon before 12,   time ::%s \n" % (presentUser,_time))

			elif (int(_time[0]) < 18 ):
				print("User ", presentUser, " usually rates during noon before 18 ")

				_file_write.write(" user %s usually rates during noon before 18,   time ::%s \n" % (presentUser,_time))
			else :
				print("User ", presentUser, " usually rates at night ")

				_file_write.write(" user %s usually rates at night,   time ::%s \n" % (presentUser,_time))
				

			#final_user.append(presentUser)
			#final_avg_rating.append(np.mean(user_rating))

			user_rating_time = []	
			presentUser = _usr 
			user_counter = 1
			#user_rating_time.append(_rating_time)
			Users.append(presentUser)

		else :

			user_rating_time.append(_rating_time)
			user_counter += 1


if (user_counter > 1) :

	print(presentUser," ",np.mean(user_rating_time))

	_hours = np.mean(user_rating_time)
	#_hours = (_hours / 60)/60  ## divding by second to get minutes  &   minutes to get hours , 
	_time = time.strftime("%H:%M:%S", time.gmtime(_hours))
	_time = _time.split(":")
	_parsed_rating_time.append(_time)

	if (int(_time[0]) < 6 ):
		print("User ", presentUser, " usually rates Early morning before 6")

		_file_write.write(" user %s usually rates Early morning before 6,   time ::%s \n" % (presentUser,_time))
	elif (int(_time[0]) < 12 ):
		print("User ", presentUser, " usually rates before noon before 12 ")
		_file_write.write(" user %s usually rates before noon before 12 ,   time ::%s \n" % (presentUser,_time))

	elif (int(_time[0]) < 18):
		print("User ", presentUser, " usually rates during noon before 18 ")
		_file_write.write(" user %s usually rates during noon before 18,   time ::%s \n" % (presentUser,_time))

	else :
		print("User ", presentUser, " usually rates at night ")
		_file_write.write(" user %s usually rates at night,   time ::%s \n" % (presentUser,_time))
		
	#_file_write.write("userID: %s  Average Rating :%s\n" %(presentUser , np.mean(user_rating_time)))
	#final_user.append(presentUser)
	#final_avg_rating.append(np.mean(user_rating))


print(type(Users) ,"  ",Users[1:5])

if(str(_searchedUserID) in Users):
	index  = Users.index(str(_searchedUserID))
	_time = _parsed_rating_time[index]
	print(index ," ",_time)
	if (int(_time[0]) < 6 ):
		print("User ", _searchedUserID, " usually rates Early morning before 6")

		_file_write.write(" user %s usually rates Early morning before 6,   time ::%s \n" % (_searchedUserID,_time))
	elif (int(_time[0]) < 12 ):
		print("User ", _searchedUserID, " usually rates before noon before 12 ")
		_file_write.write(" user %s usually rates before noon before 12 ,   time ::%s \n" % (_searchedUserID,_time))

	elif (int(_time[0]) < 18):
		print("User ", _searchedUserID, " usually rates during noon before 18 ")
		_file_write.write(" user %s usually rates during noon before 18,   time ::%s \n" % (_searchedUserID,_time))

	else :
		print("User ", _searchedUserID, " usually rates at night ")
		_file_write.write(" user %s usually rates at night,   time ::%s \n" % (_searchedUserID,_time))
		


_file_write.close()