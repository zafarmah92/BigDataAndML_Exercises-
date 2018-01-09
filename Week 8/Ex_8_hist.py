import cv2
from pyspark import SparkContext
import numpy as np

import matplotlib.pyplot as plt


sc = SparkContext("local","Simple App")

img_binary = sc.binaryFiles('/user/zfar/exercise8/castle.jpg').take(1)
img_bytes = np.asarray(bytearray(img_binary[0][1]),dtype=np.uint8)
img = cv2.imdecode(img_bytes,0)

#img = cv2.imread('dei.jpg',0)
rdd = sc.parallelize(img)
	
rdd = sc.parallelize(img).flatMap(lambda word:(word)).map(lambda item : (item , 1)).aggregateByKey(0,(lambda k,v:v+k),(lambda v,k:v+k)).collect()


new_arr = np.zeros(len(rdd))

for i in range(0, len(rdd)):
	x = rdd[i][0]
	new_arr[x] = rdd[i][1]



plt.plot(new_arr)
plt.show()
#print(img)
#cv2.imshow("file",img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

