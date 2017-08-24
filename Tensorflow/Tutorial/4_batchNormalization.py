# Author: Juan Miguel Valverde Martinez
# Date: 24 August 2017
# Youtube tutorial link: https://www.youtube.com/watch?v=bUHFDxbd5uo
# Index: http://laid.delanover.com/tensorflow-tutorial/

import tensorflow as tf
import numpy as np
import scipy.misc
from scipy.signal import convolve2d as conv
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Delete variables. Used for ipython
tf.reset_default_graph()

b_size = 60
img_height = 28
img_width = 28
classes = 10
epochs = 10

def batch_norm_wrapper(inputs, is_training, decay = 0.999):

    epsilon=0.00000001
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    mean = tf.cond(is_training,lambda: tf.nn.moments(inputs,[0])[0], lambda: tf.ones(inputs.get_shape()[-1])*pop_mean)
    var = tf.cond(is_training,lambda: tf.nn.moments(inputs,[0])[1], lambda: tf.ones(inputs.get_shape()[-1])*pop_var)
    train_mean = tf.cond(is_training,lambda:tf.assign(pop_mean,pop_mean*decay+mean*(1-decay)),lambda:tf.zeros(1))
    train_var = tf.cond(is_training,lambda:tf.assign(pop_var,pop_var*decay+var*(1-decay)),lambda:tf.zeros(1))

    with tf.control_dependencies([train_mean, train_var]):
        return tf.nn.batch_normalization(inputs,mean, var, beta, scale, epsilon)

def getModel():
        # Input: 28x28
        xi = tf.placeholder(tf.float32,[None, img_height*img_width])
        yi = tf.placeholder(tf.float32,[None, classes])
        is_training = tf.placeholder(tf.bool,[])

        with tf.variable_scope("dense1") as scope:
                W = tf.get_variable("W",shape=[img_height*img_width,1024],initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable("b",initializer=tf.zeros([1024]))

                dense = tf.matmul(xi,W)
                batched = batch_norm_wrapper(dense,is_training)
                batched+=b
                act = tf.nn.relu(batched)
        # Dense
        with tf.variable_scope("dense2") as scope:
                W = tf.get_variable("W",shape=[1024,128],initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable("b",initializer=tf.zeros([128]))

                dense = tf.matmul(act,W)
                batched = batch_norm_wrapper(dense,is_training)
                batched+=b
                act = tf.nn.relu(batched)

        with tf.variable_scope("dense3") as scope:
                W = tf.get_variable("W",shape=[128,classes],initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable("b",initializer=tf.zeros([classes]))

                dense = tf.matmul(act,W)+b


        # Prediction. We actually don't ned it
        eval_pred = tf.nn.softmax(dense)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=dense,labels=yi)
        cost = tf.reduce_mean(cross_entropy)
        train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

        return (xi,yi,is_training),train_step,cost,eval_pred


(xi,yi,is_training),train_step,cost,eval_pred = getModel()

init = tf.global_variables_initializer()
losses_list=[]

with tf.Session() as sess:
        sess.run(init)
        # Training
        for i in range(epochs):
                # Batches
                for j in range(0,mnist.train.labels.shape[0],b_size):
                #for j in range(0,120,b_size):
                        x_raw = mnist.train.images[j:j+b_size]
                        y_raw = mnist.train.labels[j:j+b_size]

                        [la,c]=sess.run([train_step,cost], feed_dict={xi: x_raw, yi: y_raw, is_training: True})
                        print("Epoch {0}/{1}. Batch: {2}/{3}. Loss: {4}".format(i+1,epochs,(j+b_size)/b_size,mnist.train.labels.shape[0]/b_size,c))

                        # To monitor the losses
                        losses_list.append(c)

        # Testing
        c=0;g=0
        for i in range(mnist.test.labels.shape[0]):
                x_raw = mnist.test.images[i:i+1] # It will just have the proper shape
                y_raw = mnist.test.labels[i:i+1]

                [pred]=sess.run([eval_pred],feed_dict={xi: x_raw, is_training: False})

                if np.argmax(y_raw)==np.argmax(pred):
                        g+=1
                c+=1

        print("Accuracy: "+str(1.0*g/c))

np.save("with",losses_list)

