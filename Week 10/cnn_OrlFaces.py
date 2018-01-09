import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.preprocessing import OneHotEncoder


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



def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    shape = [filter_size, filter_size, num_input_channels, num_filters]

    weights = new_weights(shape=shape)

    biases = new_biases(length=num_filters)
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    layer = tf.nn.relu(layer)
    return layer, weights

def pooling_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    num_features = layer_shape[1:4].num_elements()
    
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features


def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.

# We know that MNIST images are 28 pixels in each dimension.
img_size = 64

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 40

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)



layer_conv1, weights_conv1 = new_conv_layer(input=x_image,num_input_channels=num_channels,filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)

layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,num_input_channels=num_filters1, filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)

layer_flat, num_features = pooling_layer(layer_conv2)

layer_fc1 = new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=fc_size, use_relu=True)

layer_fc2 = new_fc_layer(input=layer_fc1, num_inputs=fc_size,  num_outputs=num_classes, use_relu=False)

y_pred = tf.nn.softmax(layer_fc2)

y_pred_cls = tf.argmax(y_pred, dimension=1)



cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)

cost = tf.reduce_mean(cross_entropy)


optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)


accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()
tf.summary.scalar("Accuracy:", accuracy)
tf.summary.scalar("Entropy loss:",cost)

tf.summary.histogram("weights_conv1",  weights_conv1)
tf.summary.histogram("weights_conv2",weights_conv2)

image_shaped_input = tf.reshape(x, [-1, 64, 64, 1])
tf.summary.image('input', image_shaped_input, 40)

merged = tf.summary.merge_all()


batch_size = 20
training_iters = 100
#session = tf.Session()
# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    writer = tf.summary.FileWriter('./graphs', sess.graph)
    step = 1
    # Keep training until reach max iterations
    total_batchs = x_train.shape[0]/batch_size

    print("total batches ", total_batchs)
    index = 0
    for i in range(0,training_iters):
        index = 0 
        for j in range(0,int(total_batchs)):
           # print ( j, " ", j)
            batch_x = x_train[index:index+batch_size,:]
            batch_y = y_train[index:index+batch_size,:]
            #print("batch x", batch_x.shape , "  ", "batch_y :",batch_y.shape)
            feed_dict_train = {x: batch_x,y_true: batch_y }
            sess.run(optimizer, feed_dict=feed_dict_train)

            index += batch_size
            #print("this is j ",j)
        if (i % 10 == 0) :
            _ , c , sury = sess.run([optimizer, cost , merged], feed_dict={ x: x_train , y_true : y_train }  )
            writer.add_summary(sury,i)

            acc = sess.run(accuracy, feed_dict= { x: x_test , y_true: y_test })
            print(" Epoch ",i," test accruacy ",acc  , " train cost : ", c)
        #print(i)
    """

    #while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        print("shape of batch size ",batch_y.shape)
        print("type of batch ", type(batch_y))
        print(batch_y)
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256],
                                      keep_prob: 1.}))

    """