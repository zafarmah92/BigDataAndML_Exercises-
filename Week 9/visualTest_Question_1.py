import numpy as np
import pandas as pd
import tensorflow as tf

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
	print(len(tr_x))
	
	learning_rate = 0.01
	x=tf.placeholder(tf.float32,shape=[None,4])
	y_=tf.placeholder(tf.float32,shape=[None, 3])


	W=tf.Variable(tf.zeros([4,3]), name="weights")
	b=tf.Variable(tf.zeros([3]), name="bias")
	
	logits = tf.matmul(x,W) + b
	entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_)
	loss = tf.reduce_mean(entropy)
	
	tf.contrib.deprecated.scalar_summary("Accuracy:", entropy)
	
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
	
	init = tf.global_variables_initializer()
	
	with tf.Session() as sess:
		sess.run(init)
		#n_batches = int(MNIST.train.num_examples/batch_size)
		for i in range(1000): # train the model n_epochs times
				#X_batch, Y_batch = MNIST.train.next_batch(batch_size)
			sess.run([optimizer, loss], feed_dict={x: te_x, y_:[t for t in te_y.as_matrix()]})
		writer = tf.summary.FileWriter('./graphs', sess.graph)
	
		print("Test accuracy :",sess.run(loss,feed_dict={x: te_x, y_:[t for t in te_y.as_matrix()]}))	



