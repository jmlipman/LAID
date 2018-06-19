# Author: Juan Miguel Valverde Martinez
# Date: 24 August 2017
# Youtube tutorial link: https://www.youtube.com/watch?v=9x8TR8cc9BA
# Index: http://laid.delanover.com/tensorflow-tutorial/

import numpy as np
import tensorflow as tf
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
        x = tf.placeholder("float", [None, img_width*img_height])
        y = tf.placeholder("float", [None, n_classes])

        xi = tf.reshape(x,[-1,28,28,1])

        with tf.variable_scope("conv1") as scope:
                W = tf.get_variable("W",shape=[3,3,1,32],initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable("b",initializer=tf.zeros([32]))
                l1 = tf.nn.conv2d(xi,W,strides=[1,1,1,1],padding="VALID")
                l1_b = tf.nn.bias_add(l1,b)
                l1_act = tf.nn.relu(l1_b)

        with tf.variable_scope("conv2") as scope:
                W = tf.get_variable("W",shape=[3,3,32,64],initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable("b",initializer=tf.zeros([64]))
                l2 = tf.nn.conv2d(l1_act,W,strides=[1,1,1,1],padding="VALID")
                l2_b = tf.nn.bias_add(l2,b)
                l2_act = tf.nn.relu(l2_b)

        # Maxpooling
        l3_max = tf.nn.max_pool(l2_act,[1,2,2,1],strides=[1,2,2,1],padding="VALID")


        with tf.variable_scope("dense1") as scope:
                W = tf.get_variable("W",shape=[12*12*64,128],initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable("b",initializer=tf.zeros([128]))
                l4 = tf.reshape(l3_max,[-1,12*12*64])
                l4 = tf.matmul(l4,W)
                l4_b = tf.nn.bias_add(l4,b)
                l4_act = tf.nn.relu(l4_b)


        with tf.variable_scope("dense2") as scope:
                W = tf.get_variable("W",shape=[128,n_classes],initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable("b",initializer=tf.zeros([n_classes]))
                l5 = tf.matmul(l4,W)+b


        eval_pred = tf.nn.softmax(l5)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=l5,labels=y))
        train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

        return (x,y),train_step,cost,eval_pred

def getVar(name):
        var = [v for v in tf.trainable_variables() if v.name==name+":0"][0]
        return var

(x,y),train_step,cost,eval_pred = getModel()

epochs = 1
b_size = 10

with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())


        # Training
        for i in range(epochs):
                for j in range(0,1000,b_size):
                        print("Epoch: {0}, Iteration: {1}".format(i,j))
                        x_raw = mnist.train.images[j:j+b_size]
                        y_raw = mnist.train.labels[j:j+b_size]

                        [_,c]=sess.run([train_step,cost],feed_dict={x:x_raw,y:y_raw})

        k = getVar("dense1/b").eval()

        # Testing
        gc=0;tc=0;
        for i in range(1000):
                x_raw = mnist.test.images[i:i+1]
                y_raw = mnist.test.labels[i:i+1]

                pred = sess.run(eval_pred,feed_dict={x:x_raw})

                if np.argmax(pred)==np.argmax(y_raw):
                        gc+=1
                tc+=1

        print(1.0*gc/tc)

