
from sklearn import datasets
import numpy as np 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

def prepData() :
	enc = OneHotEncoder()

	olivetti = datasets.fetch_olivetti_faces()
	X , y = olivetti.data, olivetti.target

	nb_classes = 40
	targets = np.array(y).reshape(-1)
	y_ = one_hot_targets = np.eye(nb_classes)[targets]

	#y_ = pd.DataFrame(y_)
	#X = pd.DataFrame(X)

	#frame = [X,y_]

	#df = pd.concat(frame, axis = 1)
	X_train, X_test , Y_train, Y_test = train_test_split(X, y_, test_size=0.10, random_state=42)
	#print()
	#print(np.shape(one_hot_targets))
	#y_ = pd.DataFrame(y)

	#print(np.shape(y_))
	#print(one_hot_targets)
	#

	return X_train , Y_train , X_test, Y_test
#print(y)
if __name__ == "__main__":

	tr_x , tr_y , te_x , te_y = prepData()

	print(np.shape(tr_x) , "  ", np.shape(tr_y) , " ",np.shape(te_x), "  ",np.shape(te_y))

	print(type(tr_x) , "  ", type(tr_y) , " ",type(te_x), "  ",type(te_y))

	tr_x = pd.DataFrame(tr_x)
	tr_y = pd.DataFrame(tr_y)
	te_x = pd.DataFrame(te_x)
	te_y = pd.DataFrame(te_y)


	


	x=tf.placeholder(tf.float32,shape=[None,4096] , name='X')
	y_=tf.placeholder(tf.float32,shape=[None, 40] , name='Y')

	W=tf.Variable(tf.zeros([4096,40]), name="weights")
	b=tf.Variable(tf.zeros([40]), name="bias")

	y = tf.nn.softmax(tf.matmul(x, W) + b)

	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

	train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	#sess = tf.Session()
	
	init = tf.global_variables_initializer()
	#sess.run(init)
	tf.summary.scalar("Accuracy:", accuracy)
	tf.summary.scalar("Entropy loss:",cross_entropy)
	tf.summary.histogram("weights ", W)
	tf.summary.histogram("bias",b)

	
	
	merged = tf.summary.merge_all()
		
	
	
	
	epoch=2000
	
	with tf.Session() as sess :
		sess.run(init)
		writer = tf.summary.FileWriter('./graphs', sess.graph)
          
		for step in range(0,epoch): 
			
			_,c,sury=sess.run([train_step,cross_entropy,merged], feed_dict={x: tr_x, y_:[t for t in tr_y.as_matrix()]})

			if step%100==0 : print(c)
			writer.add_summary(sury, step * epoch)
           
		
		"""
		for step in range(0,epoch):
			_,c=sess.run([train_step,cross_entropy,], feed_dict={x: te_x, y_:[t for t in te_y.as_matrix()]})
			if step%200 == 0 : print(c)
		"""
		print("Test accuracy :",sess.run(accuracy,feed_dict={x: te_x, y_:[t for t in te_y.as_matrix()]}))
		
		
          
	
    
	



