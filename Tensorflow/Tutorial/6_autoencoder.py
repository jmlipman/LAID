# Author: Juan Miguel Valverde Martinez
# Date: 5 September 2017
# Youtube tutorial link: https://www.youtube.com/watch?v=H3sg6K5iDBM
# Index: http://laid.delanover.com/tensorflow-tutorial/

# Thanks to Syam who commented on my blog a very important typo I had in the code

# Autoencoder

import numpy as np
import tensorflow as tf
import scipy.misc
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)

x_raw,y_raw = mnist.train.next_batch(40)
# Constants and variables
n_classes = 10
img_width = 28
img_height = 28

# For ipython
tf.reset_default_graph()

def getModel():
        # Network
        X = tf.placeholder("float", [None, img_width*img_height])

        layers = []

        layers.append(tf.reshape(X,[-1,28,28,1]))

        # Encoding
        with tf.variable_scope("conv1") as scope:
                W = tf.get_variable("W",shape=[4,4,1,4],initializer=tf.contrib.layers.xavier_initializer())
                layers.append(tf.nn.conv2d(layers[-1],W,strides=[1,2,2,1],padding="SAME"))
                layers.append(tf.nn.relu(layers[-1]))

        with tf.variable_scope("conv2") as scope:
                W = tf.get_variable("W",shape=[4,4,4,2],initializer=tf.contrib.layers.xavier_initializer())
                layers.append(tf.nn.conv2d(layers[-1],W,strides=[1,2,2,1],padding="SAME"))
                layers.append(tf.nn.relu(layers[-1]))

        with tf.variable_scope("conv3") as scope:
                W = tf.get_variable("W",shape=[4,4,2,1],initializer=tf.contrib.layers.xavier_initializer())
                layers.append(tf.nn.conv2d(layers[-1],W,strides=[1,2,2,1],padding="SAME"))
                layers.append(tf.nn.relu(layers[-1]))

        # Decoding
        with tf.variable_scope("conv3",reuse=True) as scope:
                W = tf.get_variable("W")
                layers.append(tf.nn.conv2d_transpose(layers[-1],W,[tf.shape(layers[-1])[0],7,7,2],strides=[1,2,2,1],padding="SAME"))
                layers.append(tf.nn.relu(layers[-1]))

        with tf.variable_scope("conv2",reuse=True) as scope:
                W = tf.get_variable("W")
                layers.append(tf.nn.conv2d_transpose(layers[-1],W,[tf.shape(layers[-5])[0],14,14,4],strides=[1,2,2,1],padding="SAME"))
                l5_act = tf.nn.relu(layers[-1])

        with tf.variable_scope("conv1",reuse=True) as scope:
                W = tf.get_variable("W")
                layers.append(tf.nn.conv2d_transpose(layers[-1],W,[tf.shape(layers[-9])[0],28,28,1],strides=[1,2,2,1],padding="SAME"))
                layers.append(tf.nn.relu(layers[-1]))

        Y = tf.reshape(layers[-1],(-1,28*28))
        cost = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(X,Y),1))
        train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

        return (X,Y),train_step,cost,layers

(x,y),train_step,cost,layers = getModel()

epochs = 1
b_size = 10

with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        test_in  = mnist.test.images[0:1]

        # Training
        for i in range(epochs):
                for j in range(0,1000,b_size):
                        print("Epoch: {0}, Iteration: {1}".format(i,j))
                        x_raw = mnist.train.images[j:j+b_size]

                        [_,c,l]=sess.run([train_step,cost,layers],feed_dict={x:x_raw})

                        [pred]=sess.run([y],feed_dict={x:test_in})


                        scipy.misc.imsave('res/'+str(j)+'.jpg', np.reshape(pred,(28,28)))
