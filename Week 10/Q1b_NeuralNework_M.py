#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
from sklearn import datasets

import numpy as np 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def singleLayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    
    print("Output layer with linear activation")
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer


def prepData() :
    enc = OneHotEncoder()

    olivetti = datasets.fetch_olivetti_faces()
    X , y = olivetti.data, olivetti.target

    nb_classes = 40
    targets = np.array(y).reshape(-1)
    y_ = one_hot_targets = np.eye(nb_classes)[targets]

    X_train, X_test , Y_train, Y_test = train_test_split(X, y_, test_size=0.10, random_state=42)

    return X_train , Y_train , X_test, Y_test



if __name__ == "__main__":


    x_train , y_train , x_test , y_test = prepData()

    x_train = pd.DataFrame(x_train)
    y_train = pd.DataFrame(y_train)
    x_test = pd.DataFrame(x_test)
    y_test = pd.DataFrame(y_test)

    x_ = x_train.values
    
    min_max_scaler = preprocessing.MinMaxScaler()
    
    x_scaled = min_max_scaler.fit_transform(x_)
    x_train = pd.DataFrame(x_scaled)
    
    
    x_ = x_test.values
    
    min_max_scaler = preprocessing.MinMaxScaler()   
    x_scaled = min_max_scaler.fit_transform(x_)
    x_test = pd.DataFrame(x_scaled)
    
    
 

    learning_rate = 0.01
    training_epochs = 1000
    beta = 0.01
    #batch_size = 20
    #display_step = 1


    print("x train shape :",x_train.shape, " Y train shape :",y_train.shape)
    print("x test shape :",x_test.shape, ""," Y test shape :",y_test.shape)
    # Network Parameters
    n_hidden_1 = 256 # 1st layer number of features
    n_hidden_2 = 256 # 2nd layer number of features
    n_hidden_3 = 256
    
    n_input = x_train.shape[1] # MNIST data input (img shape: 28*28)
    n_classes = y_test.shape[1] # MNIST total classes (0-9 digits)


    print("n_inputs",n_input)
    print("n_classes",n_classes)

    # Parameters

    # tf Graph input
    
    #x_ = tf.placeholder("float", [None, n_input])
    #y_ = tf.placeholder("float", [None, n_classes])

    x_=tf.placeholder(tf.float32,shape=[None,n_input] , name='X')
    y_=tf.placeholder(tf.float32,shape=[None, n_classes] , name='Y')



    # Create model

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),        
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Construct model
    pred = singleLayer_perceptron(x_, weights, biases)
    print("single layer Model made ")

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y_) + 0.001*tf.nn.l2_loss(weights['h1']) + 0.001*tf.nn.l2_loss(weights['h2'] + 0.001*tf.nn.l2_loss(weights['h3'])))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
        # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
       


    tf.summary.scalar("Accuracy:", accuracy)
    tf.summary.scalar("Entropy loss:",cost)

  

    tf.summary.histogram("weights_1", weights['h1'])
    tf.summary.histogram("weights_2",weights['h2'])
    tf.summary.histogram("weights_3",weights['h3'])
    tf.summary.histogram("weights_out",weights['out'])



    tf.summary.histogram("biases_1", biases['b1'])
    tf.summary.histogram("biases_2",biases['b2'])

    tf.summary.histogram("biases_3",biases['b2'])
    tf.summary.histogram("biases_out",biases['out'])
    
    image_shaped_input = tf.reshape(x_, [-1, 64, 64, 1])
    tf.summary.image('input', image_shaped_input, 40)


    merged = tf.summary.merge_all()

    #X_ = pd.DataFrame(x_train)
    #Y_ = pd.DataFrame(x_test)
    #batch_counter = 0
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        writer = tf.summary.FileWriter('./graphs', sess.graph)

        # Training cycle
        print("training epochs ",training_epochs)
        
        for epoch in range(training_epochs):

            #sess.run(init)

            #avg_cost = 0.
            #total_batch = int(x_train.shape[0]/batch_size)
            #print("total batches ",total_batch) 
            # Loop over all batches
            #batch_counter = 0
            #for i in range(total_batch):
                
                #batch_x = x_train[batch_counter:batch_counter+batch_size,:]
                #batch_y = y_train[batch_counter:batch_counter+batch_size,:]
                #print("this is batch X ",batch_x)
                #print()
                #batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                #print("batch_x",batch_x.shape)
                #print("batch_y",batch_y.shape)
            _, c , sury = sess.run([optimizer, cost, merged], feed_dict={x_: x_train, y_:[t for t in y_train.as_matrix()] })
            writer.add_summary(sury,epoch)
           
                
                #batch_counter += batch_size
                # Compute average loss
                #avg_cost += c / total_batch
            # Display logs per epoch step
            #if epoch % display_step == 0:
            if epoch%100==0 :
                print(c)
        
                print("Test accuracy :",sess.run(accuracy,feed_dict={x_: x_test, y_:[t for t in y_test.as_matrix()]}))
        
                #print("Epoch:", '%04d' % (epoch+1), "cost=","{:.9f}".format(c))
        print("Optimization Finished!")

        # Test model
        

