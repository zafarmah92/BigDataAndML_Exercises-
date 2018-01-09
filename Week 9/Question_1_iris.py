import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing


def prep_data(train_siz=135, test_siz=15):
    cols = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']  
    iris_df = pd.read_csv('iris.data', header=None, names=cols)
    
    
    class_name = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    iris_df['iclass'] = [class_name.index(class_str) 
            for class_str in iris_df['class'].values] 
                         
                         
    #print(iris_df)
    data_len = len(iris_df)
    orig = np.arange(data_len)
    perm = np.copy(orig)
    np.random.shuffle(perm)
    iris = iris_df[['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'iclass']].values
    
    #print(orig ,"  ",perm )
    iris[orig, :] = iris[perm, :]
    
    label = np.zeros((data_len, 3), dtype=np.float32)
    #print(label)                     
    
    for i in range(data_len):
        #print(int(iris[i, -1]))
        iclass = int(iris[i, -1])
        label[i][iclass] = 1.0

    #print(label)
    trX = iris[:train_siz, :-1]
    teX = iris[train_siz:, :-1]
    trY = label[:train_siz, :]
    teY = label[train_siz:, :]
                         
    return trX, trY, teX, teY
    
    
    
    
def linear_model(X, w, b):
    output = tf.matmul(X, w) + b
    return output
    
    
    
    
if __name__ == "__main__":


	tr_x, tr_y, te_x, te_y = prep_data()
	tr_x = pd.DataFrame(tr_x)
	tr_y = pd.DataFrame(tr_y)
	te_x = pd.DataFrame(te_x)
	te_y = pd.DataFrame(te_y)
	
	x_ = tr_x.values
	
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(x_)
	tr_x = pd.DataFrame(x_scaled)
	
	
	x_ = te_x.values
	min_max_scaler = preprocessing.MinMaxScaler()	
	x_scaled = min_max_scaler.fit_transform(x_)
	te_x = pd.DataFrame(x_scaled)
	
	
	min_max_scaler = preprocessing.MinMaxScaler()

	
	print(len(tr_x))
	x=tf.placeholder(tf.float32,shape=[None,4] , name='X')
	y_=tf.placeholder(tf.float32,shape=[None, 3] , name='Y')


	W=tf.Variable(tf.zeros([4,3]), name="weights")
	b=tf.Variable(tf.zeros([3]), name="bias")

	y = tf.nn.softmax(tf.matmul(x, W) + b)

	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
	# Variables

	train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	#sess = tf.InteractiveSession()
	init = tf.global_variables_initializer()
	#sess.run(init)
	
	#tf.contrib.deprecated.histogram_summary("Accuracy:", accuracy)
	
	#tf.summary.scalar_summary("Accuracy:", correct_prediction)
	tf.summary.scalar("Accuracy:", accuracy)
	tf.summary.scalar("Entropy loss:",cross_entropy)
	tf.summary.histogram("weights ", W)
	tf.summary.histogram("bias",b)
	
	merged = tf.summary.merge_all()
		
	
	
	#tf.summary.histogram_summary('softmax', y)
	
	
	#number of interations
	epoch=5000
	
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
		
		
          
	
    
