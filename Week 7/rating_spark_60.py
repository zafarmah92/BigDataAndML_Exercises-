from pyspark import SparkContext
import numpy as np

sc = SparkContext("local","Simple App")
 
logData = sc.textFile("/home/zfar/ml-10M100K/ratings.dat")

extracted_userData = logData.map(lambda x:x.split('::')).map(lambda y:(y[0],float(y[2]))).collect()

print(len(extracted_userData))
print(extracted_userData[1:5])

mat = np.matrix(extracted_userData)

Users = []
user_rating = []
presentUser = 0
user_counter = 0

final_user =[]
final_avg_rating = []

_file_write = open("60_user_rating.txt","w")

for i in range(0,len(extracted_userData)):

	_usr = mat[i,0]
	_rating = mat[i,1]

	if (i == 0):
	
		presentUser = _usr
		user_counter = 1
		Users.append(presentUser)
		user_rating.append(float(_rating))
	
	else :
		if (presentUser != _usr):

			if (user_counter < 60) :
				#print(presentUser," ",len(user_rating), " ",type(user_rating)," ",user_rating)
				print(presentUser," ",np.mean(user_rating))
				_file_write.write("userID: %s  Average Rating :%s\n" %(presentUser , np.mean(user_rating)))
				final_user.append(presentUser)
				final_avg_rating.append(np.mean(user_rating))

			user_rating = []	
			presentUser = _usr 
			user_counter = 1
			user_rating.append(float(_rating))
			Users.append(_usr)

		else :

			user_rating.append(float(_rating))
			user_counter += 1


if (user_counter < 60) :

	print(presentUser," ",np.mean(user_rating))
	_file_write.write("userID: %s  Average Rating :%s\n" %(presentUser , np.mean(user_rating)))
	final_user.append(presentUser)
	final_avg_rating.append(np.mean(user_rating))


_file_write.write("\n\n\nfinal user with max rating \n\n\n")


x = np.argmax(final_avg_rating)
_userID = final_user[x]

_file_write.write("userID: %s  Average Rating :%s\n" %(_userID , np.max(final_avg_rating)))






