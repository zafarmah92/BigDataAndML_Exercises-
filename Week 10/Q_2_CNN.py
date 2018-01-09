from __future__ import print_function

import tensorflow as tf
from sklearn import datasets

import numpy as np 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

def prepData() :
    enc = OneHotEncoder()

    olivetti = datasets.fetch_olivetti_faces()
    X , y = olivetti.data, olivetti.target

    nb_classes = 40
    targets = np.array(y).reshape(-1)
    y_ = one_hot_targets = np.eye(nb_classes)[targets]

    X_train, X_test , Y_train, Y_test = train_test_split(X, y_, test_size=0.10, random_state=42)

    return X_train , Y_train , X_test, Y_test


x_train , y_train , x_test , y_test = prepData()

x_train = pd.DataFrame(x_train)
y_train = pd.DataFrame(y_train)
x_test = pd.DataFrame(x_test)
y_test = pd.DataFrame(y_test)

#x_ = x_train.values

#min_max_scaler = preprocessing.MinMaxScaler()

#x_scaled = min_max_scaler.fit_transform(x_)
#x_train = pd.DataFrame(x_scaled)


#x_ = x_test.values

#min_max_scaler = preprocessing.MinMaxScaler()   
#x_scaled = min_max_scaler.fit_transform(x_)
#x_test = pd.DataFrame(x_scaled)


# Parameters
learning_rate = 0.001
training_epochs = 2000
batch_size = 128
display_step = 10


# Network Parameters
n_input = x_train.shape[1] # MNIST data input (img shape: 28*28)
n_classes = y_test.shape[1]# MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

print(n_input , "  " ,n_classes)

# tf Graph input
x_ = tf.placeholder(tf.float32, [None, n_input])
y_ = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

#x_train = np.asmatrix(x_train)
#y_train = np.asmatrix(y_train) 

#x_test = np.asmatrix(x_test)
#y_test = np.asmatrix(y_test)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 64, 64, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 4096])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([4096, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([4096])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x_, weights, biases, keep_prob)

# Define loss and optimizer


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
#total_batches = x_train.shape[0]/10

# Launch the graph

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        loss, acc = sess.run([cost, accuracy], feed_dict={x_: np.asmatrix(x_train), y_: np.asmatrix(y_train) ,keep_prob: 1. })
        if epoch % 100 == 0:
            print(acc)
    print("Optimization Finished!")

"""
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: x_test,
                                      y: [t for t in y_test],
                                      keep_prob: 1.}))
"""