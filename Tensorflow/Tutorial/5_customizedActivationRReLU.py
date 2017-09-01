# Author: Juan Miguel Valverde Martinez
# Date: 1 September 2017
# Youtube tutorial link: https://www.youtube.com/watch?v=wj6rY8QPGl4
# Index: http://laid.delanover.com/tensorflow-tutorial/

import tensorflow as tf
import numpy as np
import random
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


def rlrelu(tensor,bounds,is_training):
        upper=bounds[0];lower=bounds[1];

        # Random value between two bounds
        my_random=tf.Variable(tf.random_uniform([])*(upper-lower)+lower)
        alpha=tf.cond(is_training,lambda:my_random,lambda:tf.Variable((1.0*upper+lower)/2,dtype=tf.float32))
        # In addition to return the result, we return my_random for initializing on each
        # iteration and alpha to check the final value used.
        return (tf.nn.relu(tensor)-tf.nn.relu(-tensor)*alpha),my_random,alpha

# Batch normalization function
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
        bounds=(3,8)

        with tf.variable_scope("dense1") as scope:
                W = tf.get_variable("W",shape=[img_height*img_width,1024],initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable("b",initializer=tf.zeros([1024]))

                dense = tf.matmul(xi,W)
                #batched = batch_norm_wrapper(dense,is_training)
                #batched+=b

                #elu = tf.contrib.keras.layers.ELU()
                #act = elu(dense+b)
                #prelu = tf.contrib.keras.layers.PReLU()
                #act = prelu(batched)

                #act = tf.nn.relu(batched)

                act,r1,a1 = rlrelu(dense+b,bounds,is_training)

        # Dense
        with tf.variable_scope("dense2") as scope:
                W = tf.get_variable("W",shape=[1024,128],initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable("b",initializer=tf.zeros([128]))

                dense = tf.matmul(act,W)
                #batched = batch_norm_wrapper(dense,is_training)
                #batched+=b

                #elu = tf.contrib.keras.layers.ELU()
                #act = elu(dense+b)
                #prelu = tf.contrib.keras.layers.PReLU()
                #act = prelu(batched)

                #act = tf.nn.relu(batched)

                act,r2,a2 = rlrelu(dense+b,bounds,is_training)

        with tf.variable_scope("dense3") as scope:
                W = tf.get_variable("W",shape=[128,classes],initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable("b",initializer=tf.zeros([classes]))

                dense = tf.matmul(act,W)+b


        # Prediction. We actually don't ned it
        eval_pred = tf.nn.softmax(dense)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=dense,labels=yi)
        cost = tf.reduce_mean(cross_entropy)
        train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

        return (xi,yi,is_training),train_step,cost,eval_pred,r1,r2,a1,a2


(xi,yi,is_training),train_step,cost,eval_pred,r1,r2,a1,a2 = getModel()

init = tf.global_variables_initializer()
losses_list=[]

with tf.Session() as sess:
        sess.run(init)
        # Training
        for i in range(epochs):
                # Batches
                for j in range(0,mnist.train.labels.shape[0],b_size):
                        x_raw = mnist.train.images[j:j+b_size]
                        y_raw = mnist.train.labels[j:j+b_size]

                        [la,c,_,_,k1,k2]=sess.run([train_step,cost,r1,r2,a1,a2], feed_dict={xi: x_raw, yi: y_raw, is_training: True})
                        sess.run(r1.initializer)
                        sess.run(r2.initializer)
                        print("Epoch {0}/{1}. Batch: {2}/{3}. Loss: {4}, {5}, {6}".format(i+1,epochs,(j+b_size)/b_size,mnist.train.labels.shape[0]/b_size,c,k1,k2))

                        #if len(losses_list)>100:
                        #       raise Exception("")

                        # To monitor the losses
                        losses_list.append(c)

        # Testing
        c=0;g=0
        for i in range(mnist.test.labels.shape[0]):
                x_raw = mnist.test.images[i:i+1] # It will just have the proper shape
                y_raw = mnist.test.labels[i:i+1]

                [pred,_,_,k1,k2]=sess.run([eval_pred,r1,r2,a1,a2],feed_dict={xi: x_raw, is_training: False})
                sess.run(r1.initializer)
                sess.run(r2.initializer)
                print("{0}, {1}".format(k1,k2))

                if np.argmax(y_raw)==np.argmax(pred):
                        g+=1
                c+=1

        print("Accuracy: "+str(1.0*g/c))
